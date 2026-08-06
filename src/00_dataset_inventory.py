"""
00_dataset_inventory.py
=======================
Catalogue for ONE project: Illegal Lake Encroachment Prediction.

The research report shortlisted 12 candidate projects; this repository covers
only the lake one, so only the 20 sources that project needs are listed here.

`used = True` marks the four sources actually downloaded and used in this EDA.
The other 16 are options named in the report that could be added later.

Outputs
-------
data/processed/dataset_inventory.csv   the 20 sources
data/processed/lakes.csv               the 6 study lakes
data/processed/study_area.csv          the single study city
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

COLS = ["dataset_id", "dataset", "provider", "provider_country", "modality",
        "theme", "access_tier", "res_m", "year_start", "year_end",
        "cadence_days", "cost", "url", "used"]

DATASETS = [
    # ------------------------- actually used -------------------------------
    ("S2", "Sentinel-2 MSI L2A", "ESA Copernicus", "EU", "Satellite",
     "Optical imagery", "open_api", 10, 2015, 2026, 5, "Free",
     "https://dataspace.copernicus.eu", True),
    ("OSM", "OpenStreetMap (Overpass API)", "OSM Foundation", "Global",
     "Non-satellite", "Vector geodata", "open_api", None, 2004, 2026, 1,
     "Free (ODbL)", "https://overpass-api.de", True),
    ("POWER", "NASA POWER (MERRA-2 / GPM)", "NASA LaRC", "USA", "Satellite",
     "Meteorology", "open_api", 55000, 1981, 2026, 1, "Free",
     "https://power.larc.nasa.gov", True),
    ("GIBS", "NASA GIBS worldview tiles", "NASA EOSDIS", "USA", "Satellite",
     "Optical imagery", "open_api", 250, 2000, 2026, 1, "Free",
     "https://gibs.earthdata.nasa.gov", True),

    # --------------------- listed, not used yet ----------------------------
    ("LANDSAT", "Landsat 5/7/8/9 Surface Reflectance", "USGS/NASA", "USA",
     "Satellite", "Optical imagery", "open_download", 30, 1984, 2026, 16,
     "Free", "https://earthexplorer.usgs.gov", False),
    ("SRTM", "NASA SRTM DEM", "NASA JPL", "USA", "Satellite", "Elevation",
     "open_download", 30, 2000, 2000, None, "Free",
     "https://earthdata.nasa.gov", False),
    ("CARTODEM", "CartoDEM v3", "ISRO / NRSC", "India", "Satellite", "Elevation",
     "open_download", 30, 2011, 2011, None, "Free (login)",
     "https://bhuvan.nrsc.gov.in", False),
    ("IMERG", "GPM IMERG precipitation", "NASA/JAXA", "USA", "Satellite",
     "Precipitation", "open_api", 10000, 2000, 2026, 0.02, "Free",
     "https://gpm.nasa.gov", False),
    ("GHSL", "Global Human Settlement Layer", "EC JRC", "EU", "Satellite",
     "Built-up / urban", "open_download", 100, 1975, 2030, 1825, "Free",
     "https://ghsl.jrc.ec.europa.eu", False),
    ("GUF", "Global Urban Footprint", "DLR", "EU", "Satellite",
     "Built-up / urban", "open_download", 12, 2011, 2012, None,
     "Free (research)", "https://www.dlr.de", False),
    ("WORLDPOP", "WorldPop gridded population", "Univ. Southampton", "UK",
     "Satellite", "Population", "open_download", 100, 2000, 2020, 365, "Free",
     "https://www.worldpop.org", False),
    ("ASCAT", "ASCAT soil moisture", "EUMETSAT", "EU", "Satellite",
     "Soil moisture", "open_download", 25000, 2007, 2026, 1, "Free",
     "https://www.eumetsat.int", False),
    ("BHUVAN", "Bhuvan LULC & soil layers", "ISRO / NRSC", "India", "Satellite",
     "Land cover / soil", "open_download", 56, 2005, 2023, 365, "Free (login)",
     "https://bhuvan.nrsc.gov.in", False),
    ("CENSUS", "Census of India 2011", "Registrar General of India", "India",
     "Non-satellite", "Demographics", "open_download", None, 2011, 2011, None,
     "Free", "https://censusindia.gov.in", False),
    ("CPCB_WQ", "CPCB / NWIC water quality (pH, DO)", "CPCB / NWIC", "India",
     "Non-satellite", "Water quality", "open_download", None, 2012, 2025, 30,
     "Free", "https://nwic.gov.in", False),
    ("IMD", "IMD rainfall records", "India Meteorological Dept.", "India",
     "Non-satellite", "Meteorology", "open_download", 25000, 1901, 2026, 1,
     "Free / paid archives", "https://mausam.imd.gov.in", False),
    ("DATAGOVIN", "data.gov.in open data catalogue", "NIC / Govt. of India",
     "India", "Non-satellite", "Open data portal", "open_api", None, 2012,
     2026, 1, "Free", "https://data.gov.in", False),
    ("NWA", "National Wetland Atlas", "SAC / ISRO", "India", "Non-satellite",
     "Waterbody register", "open_download", None, 2011, 2011, None, "Free",
     "https://vedas.sac.gov.in", False),
    ("NEWS", "News / media scraping", "Various outlets", "Global",
     "Non-satellite", "Unstructured text", "scrape", None, 2000, 2026, 1,
     "Free / paywalled", "https://news.google.com", False),
    ("OPENMETEO", "Open-Meteo reanalysis archive", "Open-Meteo", "EU",
     "Non-satellite", "Meteorology", "open_api", 11000, 1940, 2026, 1,
     "Free (CC-BY)", "https://open-meteo.com", False),
]

datasets = pd.DataFrame(DATASETS, columns=COLS)

# --------------------------------------------------------------------------
# The single study area
# --------------------------------------------------------------------------
study_area = pd.DataFrame([
    ("Bengaluru", "Karnataka", 12.9716, 77.5946,
     "Lake belt - all six study lakes sit inside a 22 x 17 km box"),
], columns=["city", "state", "lat", "lon", "role"])

# --------------------------------------------------------------------------
# The six study lakes
# --------------------------------------------------------------------------
LAKES = [
    ("Bellandur Lake", "Bellandur Lake", 12.9366, 77.6693, 360,
     "Largest in Bengaluru; chronic froth & encroachment"),
    ("Varthur Lake",   "Varthur Lake",   12.9403, 77.7418, 180,
     "Downstream of Bellandur; heavy encroachment"),
    ("Hebbal Lake",    "Hebbal Lake",    13.0450, 77.5910,  75,
     "North Bengaluru; restored"),
    ("Madiwala Lake",  "Madiwala Lake",  12.9165, 77.6180,  45,
     "South; urban pressure"),
    ("Ulsoor Lake",    "Halasuru lake",  12.9820, 77.6210,  50,
     "Central; fully urbanised catchment (OSM uses the Kannada name Halasuru)"),
    ("Sankey Tank",    "Sankey Tank",    13.0070, 77.5730,  15,
     "Small central tank; stable"),
]
lakes = pd.DataFrame(
    LAKES, columns=["lake", "osm_name", "lat", "lon", "nominal_area_ha", "note"])


if __name__ == "__main__":
    datasets.to_csv(PROC / "dataset_inventory.csv", index=False)
    study_area.to_csv(PROC / "study_area.csv", index=False)
    lakes.to_csv(PROC / "lakes.csv", index=False)

    print("Project: Illegal Lake Encroachment Prediction")
    print(f"  sources        : {len(datasets)}")
    print(f"    used now     : {int(datasets.used.sum())}")
    print(f"    not used yet : {int((~datasets.used).sum())}")
    print(f"    satellite    : {(datasets.modality=='Satellite').sum()}")
    print(f"    non-satellite: {(datasets.modality=='Non-satellite').sum()}")
    print(f"  study city     : {study_area.city.iloc[0]}")
    print(f"  lakes          : {len(lakes)}")
