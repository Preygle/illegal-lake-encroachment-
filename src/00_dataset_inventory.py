"""
00_dataset_inventory.py
=======================
Builds the structured dataset inventory for the Predictive-DA project.

Every data source named in `deep-research-report.md` is catalogued here with its
provider, modality (satellite / non-satellite), access tier, spatial resolution,
temporal coverage and the projects that consume it.

Outputs
-------
data/processed/dataset_inventory.csv      one row per dataset
data/processed/project_dataset_edges.csv  project <-> dataset usage edge list
data/processed/projects.csv               the 12 candidate projects + scores
data/processed/study_areas.csv            geographic footprint of the work
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# 1. The 12 candidate projects, as scored in the research report
# --------------------------------------------------------------------------
PROJECTS = [
    # key, title, uniqueness, label_availability, weeks, difficulty, feasible
    ("lake",        "Illegal Lake Encroachment",      9, "Derived from imagery",   3.0, "Hard",      "Yes"),
    ("school",      "School Infrastructure Failure",  6, "Poor / none",            4.5, "Very Hard", "No"),
    ("transformer", "Transformer Failure",            6, "Very poor",              4.5, "Very Hard", "No"),
    ("treefall",    "Urban Tree Fall",                8, "Poor (manual)",          3.5, "Very Hard", "No"),
    ("pothole",     "Pothole Formation",              6, "Moderate (complaints)",  2.0, "Moderate",  "Yes"),
    ("oxygen",      "Hospital Oxygen Demand",         5, "Proxy (case counts)",    2.0, "Moderate",  "Marginal"),
    ("garbage",     "Illegal Garbage Dump",           8, "Low (complaints/manual)", 2.5, "Moderate", "Yes"),
    ("heatwave",    "Urban Heatwave Mortality Risk",  5, "Poor (few labels)",      2.0, "Moderate",  "No"),
    ("sewer",       "Sewer Blockage / Drain Choke",   6, "Moderate (complaints)",  2.5, "Moderate",  "Yes"),
    ("streetlight", "Streetlight Failure",            5, "None",                   4.5, "Very Hard", "No"),
    ("bridge",      "Bridge Corrosion Risk",          5, "None",                   4.5, "Very Hard", "No"),
    ("railxing",    "Rail Crossing Accident Risk",    4, "Very limited",           3.5, "Hard",      "No"),
]

projects = pd.DataFrame(
    PROJECTS,
    columns=["project_key", "project", "uniqueness", "label_availability",
             "collection_weeks", "difficulty", "feasible_1sem"],
)

# Ordinal encodings that make the charts sortable / colourable
projects["label_score"] = projects["label_availability"].map({
    "Derived from imagery": 3, "Moderate (complaints)": 3, "Proxy (case counts)": 2,
    "Low (complaints/manual)": 2, "Poor (manual)": 1, "Poor / none": 1,
    "Poor (few labels)": 1, "Very limited": 1, "Very poor": 0, "None": 0,
})
projects["difficulty_score"] = projects["difficulty"].map(
    {"Moderate": 1, "Hard": 2, "Very Hard": 3})
projects["feasible_score"] = projects["feasible_1sem"].map(
    {"Yes": 2, "Marginal": 1, "No": 0})

# --------------------------------------------------------------------------
# 2. Dataset inventory
#    modality      : Satellite | Non-satellite
#    access_tier   : open_api | open_download | scrape | foi_rti | restricted | commercial
#    res_m         : nominal spatial resolution in metres (NaN = non-spatial / vector)
# --------------------------------------------------------------------------
COLS = ["dataset_id", "dataset", "provider", "provider_country", "modality", "theme",
        "access_tier", "res_m", "year_start", "year_end", "cadence_days",
        "cost", "url", "used_by"]

DATASETS = [
    # ---------------------------- SATELLITE / EO ---------------------------
    ("S2",      "Sentinel-2 MSI L2A", "ESA Copernicus", "EU", "Satellite", "Optical imagery",
     "open_api", 10, 2015, 2026, 5, "Free",
     "https://dataspace.copernicus.eu", "lake;garbage;treefall"),
    ("LANDSAT", "Landsat 5/7/8/9 Surface Reflectance", "USGS/NASA", "USA", "Satellite", "Optical imagery",
     "open_download", 30, 1984, 2026, 16, "Free",
     "https://earthexplorer.usgs.gov", "lake;heatwave"),
    ("S1",      "Sentinel-1 SAR GRD", "ESA Copernicus", "EU", "Satellite", "Radar / flood extent",
     "open_api", 10, 2014, 2026, 6, "Free",
     "https://dataspace.copernicus.eu", "sewer"),
    ("SRTM",    "NASA SRTM DEM", "NASA JPL", "USA", "Satellite", "Elevation",
     "open_download", 30, 2000, 2000, None, "Free",
     "https://earthdata.nasa.gov", "lake;sewer"),
    ("CARTODEM", "CartoDEM v3", "ISRO / NRSC", "India", "Satellite", "Elevation",
     "open_download", 30, 2011, 2011, None, "Free (login)",
     "https://bhuvan.nrsc.gov.in", "lake"),
    ("POWER",   "NASA POWER (MERRA-2 / GPM)", "NASA LaRC", "USA", "Satellite", "Meteorology",
     "open_api", 55000, 1981, 2026, 1, "Free",
     "https://power.larc.nasa.gov", "lake;pothole;sewer;garbage;heatwave;oxygen;treefall;transformer;streetlight;bridge;school"),
    ("IMERG",   "GPM IMERG precipitation", "NASA/JAXA", "USA", "Satellite", "Precipitation",
     "open_api", 10000, 2000, 2026, 0.02, "Free",
     "https://gpm.nasa.gov", "lake;pothole;sewer"),
    ("GHSL",    "Global Human Settlement Layer", "EC JRC", "EU", "Satellite", "Built-up / urban",
     "open_download", 100, 1975, 2030, 1825, "Free",
     "https://ghsl.jrc.ec.europa.eu", "lake;sewer;garbage"),
    ("GUF",     "Global Urban Footprint", "DLR", "EU", "Satellite", "Built-up / urban",
     "open_download", 12, 2011, 2012, None, "Free (research)",
     "https://www.dlr.de", "lake"),
    ("WORLDPOP", "WorldPop gridded population", "Univ. Southampton", "UK", "Satellite", "Population",
     "open_download", 100, 2000, 2020, 365, "Free",
     "https://www.worldpop.org", "lake;garbage;heatwave"),
    ("MODIS",   "MODIS Land Cover / LST", "NASA", "USA", "Satellite", "Land cover / temperature",
     "open_api", 500, 2000, 2026, 1, "Free",
     "https://lpdaac.usgs.gov", "sewer;heatwave"),
    ("VIIRS",   "VIIRS Day-Night Band / Black Marble", "NASA/NOAA", "USA", "Satellite", "Night lights",
     "open_api", 500, 2012, 2026, 1, "Free",
     "https://blackmarble.gsfc.nasa.gov", "streetlight;heatwave"),
    ("PLANET",  "PlanetScope imagery", "Planet Labs", "USA", "Satellite", "Optical imagery",
     "commercial", 3, 2016, 2026, 1, "Paid / limited research quota",
     "https://www.planet.com", "garbage"),
    ("ASCAT",   "ASCAT soil moisture", "EUMETSAT", "EU", "Satellite", "Soil moisture",
     "open_download", 25000, 2007, 2026, 1, "Free",
     "https://www.eumetsat.int", "lake"),
    ("LIS_GLM", "NASA LIS / GOES-GLM lightning", "NASA/NOAA", "USA", "Satellite", "Lightning",
     "open_download", 10000, 1997, 2026, 1, "Free",
     "https://ghrc.nsstc.nasa.gov", "streetlight;transformer"),
    ("BHUVAN",  "Bhuvan LULC & soil layers", "ISRO / NRSC", "India", "Satellite", "Land cover / soil",
     "open_download", 56, 2005, 2023, 365, "Free (login)",
     "https://bhuvan.nrsc.gov.in", "lake;treefall;sewer"),
    ("GIBS",    "NASA GIBS worldview tiles", "NASA EOSDIS", "USA", "Satellite", "Optical imagery",
     "open_api", 250, 2000, 2026, 1, "Free",
     "https://gibs.earthdata.nasa.gov", "lake;heatwave"),

    # -------------------------- NON-SATELLITE ------------------------------
    ("OSM",     "OpenStreetMap (Overpass API)", "OSM Foundation", "Global", "Non-satellite", "Vector geodata",
     "open_api", None, 2004, 2026, 1, "Free (ODbL)",
     "https://overpass-api.de", "lake;pothole;sewer;garbage;treefall;streetlight;railxing;heatwave;bridge"),
    ("CENSUS",  "Census of India 2011", "Registrar General of India", "India", "Non-satellite", "Demographics",
     "open_download", None, 2011, 2011, None, "Free",
     "https://censusindia.gov.in", "lake;heatwave;oxygen;garbage;school"),
    ("BMC",     "Mumbai BMC CCRS complaints (~1.2M)", "Municipal Corp. of Greater Mumbai", "India", "Non-satellite", "Civic complaints",
     "scrape", None, 2015, 2026, 1, "Free",
     "https://portal.mcgm.gov.in", "pothole;sewer;garbage"),
    ("PMC",     "Pune PMC civic complaints", "Pune Municipal Corporation", "India", "Non-satellite", "Civic complaints",
     "scrape", None, 2017, 2026, 1, "Free",
     "https://www.pmc.gov.in", "pothole;garbage;treefall"),
    ("DELHI311", "Delhi / MCD grievance portal", "Municipal Corp. of Delhi", "India", "Non-satellite", "Civic complaints",
     "scrape", None, 2016, 2026, 1, "Free",
     "https://mcdonline.nic.in", "pothole;sewer"),
    ("CPCB_AQ", "CPCB air quality (AQI, PM2.5)", "Central Pollution Control Board", "India", "Non-satellite", "Air quality",
     "open_api", None, 2015, 2026, 1, "Free",
     "https://app.cpcbccr.com", "oxygen;bridge;heatwave"),
    ("CPCB_WQ", "CPCB / NWIC water quality (pH, DO)", "CPCB / NWIC", "India", "Non-satellite", "Water quality",
     "open_download", None, 2012, 2025, 30, "Free",
     "https://nwic.gov.in", "lake"),
    ("IMD",     "IMD rainfall & heatwave bulletins", "India Meteorological Dept.", "India", "Non-satellite", "Meteorology",
     "open_download", 25000, 1901, 2026, 1, "Free / paid archives",
     "https://mausam.imd.gov.in", "pothole;sewer;heatwave;lake;treefall"),
    ("UDISE",   "UDISE+ / AISHE school records", "Ministry of Education", "India", "Non-satellite", "Education infrastructure",
     "restricted", None, 2012, 2025, 365, "Free (aggregate only)",
     "https://udiseplus.gov.in", "school"),
    ("MOHFW",   "MoHFW / covid19india case counts", "Min. of Health & Family Welfare", "India", "Non-satellite", "Public health",
     "open_api", None, 2020, 2023, 1, "Free",
     "https://data.covid19india.org", "oxygen"),
    ("IDSP",    "IDSP dengue / malaria surveillance", "NCDC", "India", "Non-satellite", "Public health",
     "open_download", None, 2009, 2026, 7, "Free",
     "https://idsp.nic.in", "oxygen"),
    ("NFHS",    "NFHS-5 health survey", "IIPS / MoHFW", "India", "Non-satellite", "Public health",
     "open_download", None, 2019, 2021, None, "Free",
     "https://rchiips.org/nfhs", "heatwave"),
    ("DATAGOVIN", "data.gov.in open data catalogue", "NIC / Govt. of India", "India", "Non-satellite", "Open data portal",
     "open_api", None, 2012, 2026, 1, "Free",
     "https://data.gov.in", "lake;pothole;railxing;school;oxygen"),
    ("NWA",     "National Wetland Atlas", "SAC / ISRO", "India", "Non-satellite", "Waterbody register",
     "open_download", None, 2011, 2011, None, "Free",
     "https://vedas.sac.gov.in", "lake"),
    ("TREECEN", "Delhi Tree Census 2015-16", "Delhi Forest Dept.", "India", "Non-satellite", "Tree inventory",
     "foi_rti", None, 2015, 2016, None, "Free on request",
     "https://forest.delhi.gov.in", "treefall"),
    ("ITREE",   "i-Tree species database", "USDA Forest Service", "USA", "Non-satellite", "Tree inventory",
     "open_download", None, 2006, 2026, None, "Free",
     "https://www.itreetools.org", "treefall"),
    ("IR_ACC",  "Indian Railways accident summaries", "Ministry of Railways", "India", "Non-satellite", "Accident records",
     "open_download", None, 2010, 2025, 365, "Free (aggregate PDF)",
     "https://indianrailways.gov.in", "railxing"),
    ("LOKSABHA", "Lok Sabha Q&A written answers", "Parliament of India", "India", "Non-satellite", "Accident records",
     "scrape", None, 2004, 2026, None, "Free",
     "https://sansad.in", "railxing"),
    ("PWD",     "State PWD / IRC bridge registers", "State PWDs / IRC", "India", "Non-satellite", "Asset inventory",
     "foi_rti", None, 1990, 2025, None, "On request",
     "https://morth.nic.in", "bridge"),
    ("GMAPS",   "Google Maps Traffic API", "Google", "USA", "Non-satellite", "Traffic",
     "commercial", None, 2015, 2026, 0.01, "Paid",
     "https://developers.google.com/maps", "pothole"),
    ("NOAA",    "NOAA Storm Events database", "NOAA", "USA", "Non-satellite", "Meteorology",
     "open_download", None, 1950, 2026, 1, "Free",
     "https://www.ncdc.noaa.gov", "transformer;bridge"),
    ("KAGGLE_DGA", "Kaggle transformer DGA datasets", "Kaggle community", "Global", "Non-satellite", "Sensor / lab",
     "open_download", None, 2018, 2024, None, "Free",
     "https://www.kaggle.com", "transformer"),
    ("RTI",     "RTI / FOI requests", "Various Indian agencies", "India", "Non-satellite", "Administrative records",
     "foi_rti", None, 2005, 2026, None, "INR 10 per request",
     "https://rtionline.gov.in", "school;transformer;treefall;bridge;pothole"),
    ("NEWS",    "News / media scraping", "Various outlets", "Global", "Non-satellite", "Unstructured text",
     "scrape", None, 2000, 2026, 1, "Free / paywalled",
     "https://news.google.com", "treefall;garbage;bridge;lake"),
    ("TIMETBL", "Indian Railways timetables", "Ministry of Railways", "India", "Non-satellite", "Transport schedules",
     "scrape", None, 2010, 2026, 365, "Free",
     "https://indianrail.gov.in", "railxing"),
    ("AADT",    "Highway AADT / DPR traffic counts", "NHAI / State highway boards", "India", "Non-satellite", "Traffic",
     "restricted", None, 2010, 2025, 365, "Free (scattered PDFs)",
     "https://nhai.gov.in", "bridge;railxing"),
    ("OPENMETEO", "Open-Meteo reanalysis archive", "Open-Meteo", "EU", "Non-satellite", "Meteorology",
     "open_api", 11000, 1940, 2026, 1, "Free (CC-BY)",
     "https://open-meteo.com", "pothole;sewer;lake;heatwave"),
]

datasets = pd.DataFrame(DATASETS, columns=COLS)

# --------------------------------------------------------------------------
# 3. Long-form project <-> dataset edge list
# --------------------------------------------------------------------------
edges = (
    datasets.assign(project_key=datasets["used_by"].str.split(";"))
            .explode("project_key")
            .loc[:, ["dataset_id", "dataset", "provider", "modality", "theme",
                     "access_tier", "project_key"]]
            .merge(projects[["project_key", "project"]], on="project_key", how="left")
)

# --------------------------------------------------------------------------
# 4. Study areas actually named across the report
# --------------------------------------------------------------------------
STUDY_AREAS = [
    # city, state, lat, lon, projects that name it, role
    ("Bengaluru", "Karnataka",   12.9716, 77.5946, "lake;pothole;sewer;garbage", "Primary case study - lake belt"),
    ("Mumbai",    "Maharashtra", 19.0760, 72.8777, "pothole;sewer;garbage",      "BMC CCRS complaint corpus"),
    ("Pune",      "Maharashtra", 18.5204, 73.8567, "pothole;garbage;treefall",   "PMC complaint portal"),
    ("Delhi",     "Delhi",       28.6139, 77.2090, "pothole;treefall;transformer", "Tree census + discom bulletins"),
    ("Chennai",   "Tamil Nadu",  13.0827, 80.2707, "pothole;sewer;lake",         "Groundwater / flood context"),
    ("Hyderabad", "Telangana",   17.3850, 78.4867, "lake;garbage",               "Lake encroachment comparison"),
    ("Kolkata",   "West Bengal", 22.5726, 88.3639, "sewer;heatwave",             "Drainage & heat context"),
    ("Ahmedabad", "Gujarat",     23.0225, 72.5714, "heatwave",                   "Heat action plan reference"),
]
study_areas = pd.DataFrame(
    STUDY_AREAS,
    columns=["city", "state", "lat", "lon", "used_by", "role"])

# --------------------------------------------------------------------------
# 5. The lakes used for the satellite deep-dive (Bengaluru)
# --------------------------------------------------------------------------
LAKES = [
    # name, osm_name (exact OSM `name` tag), lat, lon, nominal_area_ha, note
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

# --------------------------------------------------------------------------
if __name__ == "__main__":
    datasets.to_csv(PROC / "dataset_inventory.csv", index=False)
    edges.to_csv(PROC / "project_dataset_edges.csv", index=False)
    projects.to_csv(PROC / "projects.csv", index=False)
    study_areas.to_csv(PROC / "study_areas.csv", index=False)
    lakes.to_csv(PROC / "lakes.csv", index=False)

    print(f"datasets        : {len(datasets):3d} rows -> dataset_inventory.csv")
    print(f"  satellite     : {(datasets.modality=='Satellite').sum()}")
    print(f"  non-satellite : {(datasets.modality=='Non-satellite').sum()}")
    print(f"edges           : {len(edges):3d} rows -> project_dataset_edges.csv")
    print(f"projects        : {len(projects):3d} rows -> projects.csv")
    print(f"study areas     : {len(study_areas):3d} rows -> study_areas.csv")
    print(f"lakes           : {len(lakes):3d} rows -> lakes.csv")
    print()
    print(datasets.groupby(["modality", "access_tier"]).size().to_string())
