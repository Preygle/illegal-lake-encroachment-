"""
14_train_unet.py
================
Trains the four-class U-Net under nested leave-one-lake-out and writes
out-of-fold land-cover maps.

Every lake is predicted by the fold in which it was held out, so the maps that
feed the geo-fence and the risk model were produced by a model that never saw
that lake. Anything else would quietly launder training data into the result
table.

    python src/14_train_unet.py                     # all six folds
    python src/14_train_unet.py --folds 0           # one fold, to check the setup
    python src/14_train_unet.py --shuffle-labels    # negative control
    python src/14_train_unet.py --device cuda       # GPU

Writes outputs/dl/unet/{per_fold.csv, confusion_*.txt, fold<k>_best.pt,
summary.json} and outputs/dl/pred/landcover_<year>.npy (out-of-fold).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dl import common as C
from dl import dataset as D
from dl import metrics as M
from dl.losses import SegLoss
from dl.unet import TemperatureScaling, UNet

OUT = C.DL / "unet"
PRED = C.DL / "pred"


def run_tag(args) -> str:
    """Suffix that keeps every negative-control output apart from the real run."""
    return "_shuffled" if args.shuffle_labels else ""


# ==========================================================================
# Inference
# ==========================================================================
@torch.no_grad()
def predict_region(model, store, lake, year, mean, std, device,
                   temperature: float = 1.0):
    """
    Overlap-tiled inference over one lake's bounding box.

    Logits from overlapping tiles are averaged before the softmax, so seams
    between tiles do not show up as a line of misclassified pixels.
    """
    model.eval()
    _, shape, _, _ = C.grid()
    region = store.rasters[lake]["region"]
    r0, r1, c0, c1 = D._bbox(region)
    pad = D.PATCH
    r0, c0 = max(0, r0 - pad // 2), max(0, c0 - pad // 2)
    r1, c1 = min(shape[0], r1 + pad // 2), min(shape[1], c1 + pad // 2)

    chan = store.chan[year]
    acc = np.zeros((C.N_CLASSES, shape[0], shape[1]), "float32")
    hits = np.zeros(shape, "float32")

    mu = mean.reshape(-1, 1, 1)
    sd = std.reshape(-1, 1, 1)
    for rs in D._starts(r0, r1, shape[0]):
        for cs in D._starts(c0, c1, shape[1]):
            tile = (chan[:, rs:rs + D.PATCH, cs:cs + D.PATCH] - mu) / sd
            x = torch.from_numpy(np.ascontiguousarray(tile))[None].to(device)
            logits = model(x)[0].cpu().numpy() / max(temperature, 1e-6)
            acc[:, rs:rs + D.PATCH, cs:cs + D.PATCH] += logits
            hits[rs:rs + D.PATCH, cs:cs + D.PATCH] += 1

    ok = hits > 0
    acc[:, ok] /= hits[ok]
    e = np.exp(acc - acc.max(axis=0, keepdims=True))
    probs = e / np.maximum(e.sum(axis=0, keepdims=True), 1e-12)
    pred = probs.argmax(axis=0).astype("uint8")
    pred[~ok] = C.IGNORE
    return pred, probs


@torch.no_grad()
def collect_logits(model, loader, device):
    """Flatten a loader's logits and targets, dropping IGNORE. For calibration."""
    model.eval()
    L, T = [], []
    for x, m in loader:
        logits = model(x.to(device))
        valid = m != C.IGNORE
        if not valid.any():
            continue
        lg = logits.permute(0, 2, 3, 1).reshape(-1, C.N_CLASSES)
        tg = m.reshape(-1)
        keep = valid.reshape(-1)
        L.append(lg[keep.to(lg.device)].cpu())
        T.append(tg[keep])
    if not L:
        return None, None
    return torch.cat(L), torch.cat(T)


# ==========================================================================
# One fold
# ==========================================================================
def train_fold(fold, store, index, args, device):
    C.set_seed(args.seed + fold["fold"])
    train_lakes, val_lake, test_lake = fold["train"], fold["val"], fold["test"]

    # Statistics from the training lakes only. Including the validation or test
    # lake here would be the pre-split preprocessing leak.
    mean, std = D.fit_norm(store, index, train_lakes)
    counts = D.class_counts(store, index, train_lakes)
    weights = D.class_weights(counts)

    g = torch.Generator()
    g.manual_seed(args.seed + fold["fold"])
    common = dict(num_workers=0, worker_init_fn=C.seed_worker, generator=g)

    tr = D.LakePatches(store, index, train_lakes, mean, std, augment=True,
                       shuffle_labels=args.shuffle_labels, seed=args.seed)
    va = D.LakePatches(store, index, [val_lake], mean, std, augment=False,
                       shuffle_labels=args.shuffle_labels, seed=args.seed + 1)
    tr_dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                       drop_last=len(tr) > args.batch_size, **common)
    va_dl = DataLoader(va, batch_size=args.batch_size, shuffle=False, **common)

    groups = 8 if args.groupnorm or args.batch_size <= 4 else None
    model = UNet(base=args.base, groups=groups).to(device)
    crit = SegLoss(class_weight=weights, dice_weight=args.dice_weight,
                   focal=args.focal)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5)

    print(f"    {len(tr)} train / {len(va)} val patches, "
          f"{model.n_params:,d} params, weights="
          f"[{', '.join(f'{w:.2f}' for w in weights)}]")

    # Best weights are kept in memory and written once at the end of the fold.
    # Re-saving to the same path on every improvement trips Windows error 1224
    # (the file still has a mapped section open from the previous write), and
    # it is slower besides - this model is only ~2 MB of parameters.
    #
    # The negative control writes to its own filename. It used to share
    # fold<k>_best.pt with the real run, so running the control after the real
    # model silently replaced every real checkpoint with a model trained on
    # scrambled labels - the reported scores stayed correct while the saved
    # weights became useless.
    ckpt = OUT / f"fold{fold['fold']}_best{run_tag(args)}.pt"
    best_state = None
    best, bad, best_epoch = float("inf"), 0, -1
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for x, m in tr_dl:
            x, m = x.to(device), m.to(device)
            opt.zero_grad()
            loss = crit(model(x), m)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(x)
        tr_loss = tot / max(len(tr), 1)

        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for x, m in va_dl:
                x, m = x.to(device), m.to(device)
                vtot += float(crit(model(x), m).detach()) * len(x)
        va_loss = vtot / max(len(va), 1)
        sched.step(va_loss)

        if va_loss < best - 1e-4:
            best, bad, best_epoch = va_loss, 0, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1

        if epoch % 10 == 0 or bad >= args.patience:
            print(f"      epoch {epoch:3d}  train {tr_loss:.4f}  "
                  f"val {va_loss:.4f}  lr {opt.param_groups[0]['lr']:.1e}"
                  f"{'  *' if best_epoch == epoch else ''}")
        if bad >= args.patience:
            print(f"      early stop at {epoch} (best {best_epoch}, {best:.4f})")
            break

    # Always evaluate the best weights, never whatever the last epoch left behind.
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, ckpt)

    temp = 1.0
    lg, tg = collect_logits(model, va_dl, device)
    if lg is not None and len(tg) > 100:
        ts = TemperatureScaling().to("cpu")
        ts.fit(lg.float(), tg.long())
        temp = ts.temperature
    print(f"      temperature {temp:.3f} (fitted on {val_lake})")

    # Out-of-fold prediction for the held-out lake, every year.
    region = store.rasters[test_lake]["region"]
    cm = np.zeros((C.N_CLASSES, C.N_CLASSES), "int64")
    probs_flat, true_flat, maps = [], [], {}
    for year in C.YEARS:
        pred, probs = predict_region(model, store, test_lake, year,
                                     mean, std, device, temp)
        maps[year] = pred
        lab = store.label_for(test_lake, year)
        valid = region & (lab != C.IGNORE) & (pred != C.IGNORE)
        if valid.any():
            cm += M.confusion(pred[valid], lab[valid])
            probs_flat.append(probs[:, valid].T)
            true_flat.append(lab[valid])

    res = M.metrics_from_cm(cm)
    if probs_flat:
        res["ece"] = M.expected_calibration_error(
            np.concatenate(probs_flat), np.concatenate(true_flat))
    res.update(fold=fold["fold"], test_lake=test_lake, val_lake=val_lake,
               temperature=temp, best_epoch=best_epoch, best_val_loss=best,
               n_train_patches=len(tr), n_params=model.n_params)
    return res, cm, maps


# ==========================================================================
def main(args):
    device = torch.device(args.device) if args.device else C.get_device()
    OUT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    C.set_seed(args.seed)

    print(f"device: {device}")
    print("[1/3] loading scenes and labels")
    store = D.PatchStore()

    print("[2/3] patch index")
    index = D.build_index(store.rasters)

    print("[3/3] folds")
    folds = C.folds()
    if args.folds:
        folds = [f for f in folds if f["fold"] in args.folds]

    _, shape, _, _ = C.grid()
    oof = {y: np.full(shape, C.IGNORE, "uint8") for y in C.YEARS}
    rows = []
    for f in folds:
        t0 = time.time()
        print(f"\n  fold {f['fold']}: test={f['test']}  val={f['val']}  "
              f"train={len(f['train'])} lakes")
        res, cm, maps = train_fold(f, store, index, args, device)
        rows.append(res)
        (OUT / f"confusion_{f['test'].replace(' ', '_')}{run_tag(args)}.txt").write_text(
            M.format_cm(cm), encoding="utf-8")

        region = store.rasters[f["test"]]["region"]
        for y, pred in maps.items():
            put = region & (pred != C.IGNORE)
            oof[y][put] = pred[put]

        print(f"    macro-F1 {res['macro_f1']:.3f}  kappa {res['kappa']:.3f}  "
              f"IoU water {res['iou_water']:.3f} veg {res['iou_vegetation']:.3f} "
              f"bare {res['iou_bare']:.3f} built {res['iou_built']:.3f}  "
              f"({time.time()-t0:.0f}s)")
        print(f"    water->veg {res['water_as_veg']:.3f}  "
              f"veg->water {res['veg_as_water']:.3f}")

    df = pd.DataFrame(rows)
    tag = run_tag(args)
    df.to_csv(OUT / f"per_fold{tag}.csv", index=False)

    if not args.shuffle_labels and len(folds) == len(C.folds()):
        for y, arr in oof.items():
            np.save(PRED / f"landcover_{y}.npy", arr)
        print(f"\n  out-of-fold land-cover maps -> {PRED}")

    print(f"\n=== held-out means over {len(df)} lakes ===")
    print(f"  macro-F1 {df.macro_f1.mean():.3f} "
          f"[{df.macro_f1.min():.3f}-{df.macro_f1.max():.3f}]   "
          f"kappa {df.kappa.mean():.3f}")
    for n in C.CLASS_NAMES:
        print(f"    IoU {n:11s} {df['iou_' + n].mean():.3f}")
    if "ece" in df:
        print(f"  ECE {df.ece.mean():.4f}")

    summary = {"mean_macro_f1": float(df.macro_f1.mean()),
               "per_lake": dict(zip(df.test_lake, df.macro_f1.round(4))),
               "shuffle_control": bool(args.shuffle_labels)}

    rf_path = C.DL / "rf" / "summary.json"
    if rf_path.exists() and not args.shuffle_labels:
        rf = json.loads(rf_path.read_text(encoding="utf-8"))
        rf_f1 = rf["rf_mean_macro_f1"]
        delta = summary["mean_macro_f1"] - rf_f1
        summary.update(rf_mean_macro_f1=rf_f1, delta_vs_rf=delta,
                       recommended="unet" if delta > 0.02 else "random_forest")
        print(f"\n  Random Forest {rf_f1:.3f}   U-Net {summary['mean_macro_f1']:.3f}"
              f"   delta {delta:+.3f}")
        print(f"  pre-registered rule -> recommend: {summary['recommended'].upper()}"
              + ("" if delta > 0.02
                 else "  (U-Net did not clear the 2-point margin; report it as the"
                      " deep comparison that was built and tested)"))
        try:
            from scipy.stats import wilcoxon

            per = pd.read_csv(C.DL / "rf" / "per_fold.csv")
            per = per[per.model == "random_forest"].set_index("test_lake").macro_f1
            paired = df.set_index("test_lake").macro_f1
            common_lakes = per.index.intersection(paired.index)
            if len(common_lakes) >= 5:
                st, p = wilcoxon(paired[common_lakes], per[common_lakes])
                summary["wilcoxon_p"] = float(p)
                print(f"  Wilcoxon signed-rank p={p:.3f} over {len(common_lakes)} "
                      f"paired lakes - with six pairs this test has very little "
                      f"power, so treat it as descriptive")
        except Exception as e:                                    # noqa: BLE001
            print(f"  (paired test skipped: {type(e).__name__}: {e})")

    (OUT / f"summary{tag}.json").write_text(json.dumps(summary, indent=2),
                                            encoding="utf-8")
    print(f"  -> {OUT}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--focal", action="store_true",
                    help="focal loss instead of weighted CE (try weights first)")
    ap.add_argument("--groupnorm", action="store_true",
                    help="GroupNorm instead of BatchNorm (auto when batch <= 4)")
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="negative control: permuted labels must score at chance")
    main(ap.parse_args())
