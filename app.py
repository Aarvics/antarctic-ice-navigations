import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xarray as xr
import requests
import json
import math
import re

from pathlib import Path
from scipy.spatial import cKDTree
from pyproj import Transformer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PolarSafe | Antarctica Environmental Risk",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# THEME
# ============================================================

BLUE = "#0b2d4d"
BLUE_2 = "#123f63"
BLUE_3 = "#1c587d"
LIGHT_BLUE = "#eaf6ff"
ICE_BLUE = "#dff3ff"
CYAN = "#38bdf8"
WHITE = "#ffffff"
TEXT = "#102a43"
MUTED = "#486581"
BORDER = "#c7dceb"

GREEN = "#16a34a"
YELLOW = "#f59e0b"
RED = "#dc2626"


st.markdown(
    f"""
    <style>

    .stApp {{
        background:
            linear-gradient(
                135deg,
                #eef8ff 0%,
                #ffffff 45%,
                #e7f5ff 100%
            );
        color: {TEXT};
    }}

    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }}

    section[data-testid="stSidebar"] {{
        background:
            linear-gradient(
                180deg,
                #e7f5ff 0%,
                #d7efff 50%,
                #c8e8fa 100%
            );
        border-right: 1px solid {BORDER};
    }}

    section[data-testid="stSidebar"] * {{
        color: {TEXT} !important;
    }}

    h1, h2, h3, h4 {{
        color: {BLUE} !important;
    }}

    p, label, span, div {{
        color: {TEXT};
    }}

    .stCaption {{
        color: {MUTED} !important;
    }}

    div[data-testid="stMetric"] {{
        background: rgba(255,255,255,0.85);
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 14px;
        box-shadow: 0 4px 15px rgba(18,63,99,0.06);
    }}

    div[data-testid="stMetricLabel"] {{
        color: {MUTED} !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: {BLUE} !important;
        font-weight: 700;
    }}

    .card {{
        background: rgba(255,255,255,0.90);
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 5px 20px rgba(18,63,99,0.07);
        margin-bottom: 15px;
    }}

    .section-title {{
        color: {BLUE};
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 4px;
    }}

    .section-subtitle {{
        color: {MUTED};
        font-size: 13px;
        margin-bottom: 15px;
    }}

    .risk-high {{
        background: #fee2e2;
        border: 1px solid #fecaca;
        color: #991b1b;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }}

    .risk-medium {{
        background: #fef3c7;
        border: 1px solid #fde68a;
        color: #92400e;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }}

    .risk-low {{
        background: #dcfce7;
        border: 1px solid #bbf7d0;
        color: #166534;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }}

    .data-status {{
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 8px;
    }}

    .stButton > button {{
        background: linear-gradient(
            90deg,
            #0b2d4d,
            #17618c
        );
        color: white !important;
        border: none;
        border-radius: 10px;
        font-weight: 700;
    }}

    .stButton > button:hover {{
        background: linear-gradient(
            90deg,
            #17618c,
            #38bdf8
        );
        color: white !important;
    }}

    hr {{
        border-color: {BORDER};
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

BED_MACHINE_NAME = (
    "NSIDC-0756_BedMachineAntarctica_"
    "19700101-20191001_V04.1.nc"
)

SEA_ICE_NAME = "G10010_sibt1850_v2.0.nc"

ICEBERG_URL = (
    "https://raw.githubusercontent.com/"
    "Joel-hanson/Iceberg-locations/main/api/latest.json"
)

CURRENT_BASE_URL = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/"
    "noaacwBLENDEDNRTcurrentsDaily.csv"
)

SEA_ICE_FALLBACK_URL = (
    "https://ftp.cpc.ncep.noaa.gov/hwang/sea_ice/"
    "G10010_sibt1850_v2.0.nc"
)

# Real Antarctic locations
ANTARCTIC_LOCATIONS = {
    "Rothera Research Station": (-67.568, -68.129),
    "Vernadsky Research Base": (-65.246, -64.257),
    "McMurdo Station": (-77.846, 166.668),
    "Casey Research Station": (-66.282, 110.524),
    "Dumont d'Urville Station": (-66.663, 140.002),
    "Neumayer Station III": (-70.650, -8.250),
    "Halley VI Research Station": (-75.582, -26.655),
}


# ============================================================
# FILE FINDER
# ============================================================

def find_local_file(filename):

    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "data" / filename,
        BASE_DIR / "data" / "raw" / filename,
        BASE_DIR / "data" / "raw" / "sea_ice" / filename,
        BASE_DIR / "data" / "raw" / "bedmachine" / filename,
        BASE_DIR / "bedmachine_data" / filename,
        Path("/content") / filename,
        Path("/content/bedmachine_data") / filename,
        Path("/content/data/raw/sea_ice") / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    # Recursive search inside repository
    try:
        matches = list(BASE_DIR.rglob(filename))
        if matches:
            return matches[0]
    except Exception:
        pass

    # Colab search
    try:
        content = Path("/content")

        if content.exists():
            matches = list(content.rglob(filename))

            if matches:
                return matches[0]

    except Exception:
        pass

    return None


# ============================================================
# BEDMACHINE LOADER
# ============================================================

@st.cache_data(show_spinner=False)
def load_bedmachine():

    path = find_local_file(BED_MACHINE_NAME)

    if path is None:

        return {
            "available": False,
            "message": (
                "BedMachine file was not found in the Streamlit repository. "
                "The dashboard will still run, but bathymetry-based risk "
                "will be unavailable."
            )
        }

    try:

        ds = xr.open_dataset(path)

        x = ds["x"].values
        y = ds["y"].values

        # Downsample the 500 m BedMachine grid.
        # We do NOT load the entire 13,333 x 13,333 grid.
        target = 180

        step_x = max(1, len(x) // target)
        step_y = max(1, len(y) // target)

        x_small = x[::step_x]
        y_small = y[::step_y]

        bed = ds["bed"].isel(
            x=slice(None, None, step_x),
            y=slice(None, None, step_y)
        ).values.astype(float)

        mask = ds["mask"].isel(
            x=slice(None, None, step_x),
            y=slice(None, None, step_y)
        ).values.astype(np.int8)

        errbed = ds["errbed"].isel(
            x=slice(None, None, step_x),
            y=slice(None, None, step_y)
        ).values.astype(float)

        ds.close()

        xx, yy = np.meshgrid(x_small, y_small)

        transformer = Transformer.from_crs(
            "EPSG:3031",
            "EPSG:4326",
            always_xy=True
        )

        lon, lat = transformer.transform(xx, yy)

        # Antarctica BedMachine mask:
        # 0 = ocean
        # 1 = ice-free land
        # 2 = grounded ice
        # 3 = floating ice
        # 4 = Lake Vostok

        ocean = (
            (mask == 0)
            & np.isfinite(bed)
            & (bed < 50)
        )

        # Calculate seabed slope
        dx = abs(float(np.nanmedian(np.diff(x_small))))
        dy = abs(float(np.nanmedian(np.diff(y_small))))

        bed_clean = np.where(np.isfinite(bed), bed, np.nan)

        gy, gx = np.gradient(
            bed_clean,
            dy,
            dx
        )

        slope = np.sqrt(
            np.nan_to_num(gx) ** 2
            +
            np.nan_to_num(gy) ** 2
        )

        slope = np.clip(slope, 0, 1)

        return {
            "available": True,
            "path": str(path),
            "x": x_small,
            "y": y_small,
            "lon": lon,
            "lat": lat,
            "bed": bed,
            "mask": mask,
            "errbed": errbed,
            "slope": slope,
            "ocean": ocean
        }

    except Exception as e:

        return {
            "available": False,
            "message": f"BedMachine could not be opened: {e}"
        }


# ============================================================
# SEA ICE LOADER
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_sea_ice():

    # First look for user's processed CSV
    processed = BASE_DIR / "data" / "processed" / "sea_ice"

    if processed.exists():

        csv_files = list(
            processed.glob("*.csv")
        )

        if csv_files:

            try:

                df = pd.read_csv(csv_files[0])

                possible_ice = [
                    "sea_ice_concentration",
                    "seaice_conc",
                    "F17_ICECON"
                ]

                ice_col = None

                for c in possible_ice:
                    if c in df.columns:
                        ice_col = c
                        break

                if ice_col is not None:

                    lat_col = (
                        "latitude"
                        if "latitude" in df.columns
                        else "lat"
                    )

                    lon_col = (
                        "longitude"
                        if "longitude" in df.columns
                        else "lon"
                    )

                    out = df[
                        [lat_col, lon_col, ice_col]
                    ].copy()

                    out.columns = [
                        "latitude",
                        "longitude",
                        "ice"
                    ]

                    out["ice"] = pd.to_numeric(
                        out["ice"],
                        errors="coerce"
                    )

                    # Convert 0-1 to percentage
                    if out["ice"].max() <= 1.5:
                        out["ice"] *= 100

                    out = out[
                        out["latitude"] <= -50
                    ]

                    out = out.dropna()

                    return {
                        "available": True,
                        "data": out,
                        "date": "Local processed dataset"
                    }

            except Exception:
                pass

    # Try raw G10010
    path = find_local_file(SEA_ICE_NAME)

    if path is None:

        # Download public NOAA copy automatically
        try:

            response = requests.get(
                SEA_ICE_FALLBACK_URL,
                timeout=60
            )

            response.raise_for_status()

            temp_path = (
                Path("/tmp") /
                "G10010_sibt1850_v2.0.nc"
            )

            with open(temp_path, "wb") as f:
                f.write(response.content)

            path = temp_path

        except Exception as e:

            return {
                "available": False,
                "message": f"Sea-ice data unavailable: {e}"
            }

    try:

        ds = xr.open_dataset(path)

        if "seaice_conc" in ds:

            variable = ds["seaice_conc"]

            # Latest available observation
            variable = variable.isel(time=-1)

            lat_name = "latitude"
            lon_name = "longitude"

            source_lat = ds[lat_name].values
            source_lon = ds[lon_name].values

            values = variable.values.astype(float)

            # Convert packed values if necessary
            if np.nanmax(values) > 100:
                values = values / 250.0 * 100

            elif np.nanmax(values) <= 1.5:
                values *= 100

            lat_grid, lon_grid = np.meshgrid(
                source_lat,
                source_lon,
                indexing="ij"
            )

            df = pd.DataFrame({
                "latitude": lat_grid.ravel(),
                "longitude": lon_grid.ravel(),
                "ice": values.ravel()
            })

            df = df[
                df["latitude"] <= -50
            ]

            df = df[
                (df["ice"] >= 0)
                &
                (df["ice"] <= 100)
            ]

            df = df.dropna()

            # Limit display size
            if len(df) > 8000:
                df = df.sample(
                    8000,
                    random_state=42
                )

            date = str(
                ds["time"].values[-1]
            )

            ds.close()

            return {
                "available": True,
                "data": df,
                "date": date
            }

        # Support processed NSIDC polar stereographic file
        if "F17_ICECON" in ds:

            variable = ds["F17_ICECON"]

            if "time" in variable.dims:
                variable = variable.isel(time=0)

            arr = variable.values.astype(float)

            if np.nanmax(arr) <= 1.5:
                arr *= 100

            x = ds["x"].values
            y = ds["y"].values

            xx, yy = np.meshgrid(x, y)

            transformer = Transformer.from_crs(
                "EPSG:3412",
                "EPSG:4326",
                always_xy=True
            )

            lon, lat = transformer.transform(
                xx,
                yy
            )

            df = pd.DataFrame({
                "latitude": lat.ravel(),
                "longitude": lon.ravel(),
                "ice": arr.ravel()
            })

            df = df[
                (df["latitude"] <= -50)
                &
                (df["ice"] >= 0)
                &
                (df["ice"] <= 100)
            ]

            df = df.dropna()

            if len(df) > 8000:
                df = df.sample(
                    8000,
                    random_state=42
                )

            ds.close()

            return {
                "available": True,
                "data": df,
                "date": "NSIDC Antarctic sea-ice grid"
            }

        ds.close()

        return {
            "available": False,
            "message": "Recognized NetCDF file but no sea-ice variable was found."
        }

    except Exception as e:

        return {
            "available": False,
            "message": f"Sea-ice file could not be read: {e}"
        }


# ============================================================
# ICEBERG LOADER
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def load_icebergs():

    try:

        response = requests.get(
            ICEBERG_URL,
            timeout=30
        )

        response.raise_for_status()

        raw = response.json()

        rows = []

        # Most versions of the data are date -> records
        if isinstance(raw, dict):

            if all(
                isinstance(v, list)
                for v in raw.values()
            ):

                latest_key = sorted(
                    raw.keys()
                )[-1]

                records = raw[latest_key]

            else:

                records = raw

        elif isinstance(raw, list):

            records = raw

        else:

            records = []

        for item in records:

            if not isinstance(item, dict):
                continue

            name = (
                item.get("iceberg")
                or item.get("name")
                or item.get("id")
                or "Unknown"
            )

            lat = item.get("latitude")
            lon = item.get("longitude")

            try:

                lat = float(lat)
                lon = float(lon)

            except Exception:

                continue

            if lat > -45:
                continue

            rows.append({
                "iceberg": str(name).upper(),
                "latitude": lat,
                "longitude": lon,
                "observation": item.get(
                    "recent_observation",
                    ""
                )
            })

        df = pd.DataFrame(rows)

        if len(df) == 0:

            return {
                "available": False,
                "message": "No current iceberg records returned."
            }

        return {
            "available": True,
            "data": df
        }

    except Exception as e:

        return {
            "available": False,
            "message": f"Iceberg service unavailable: {e}"
        }


# ============================================================
# OCEAN CURRENT LOADER
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def load_currents():

    try:

        query = (
            "u_current[(last)][(-80):4:(-55)]"
            "[(-180):8:(180)],"
            "v_current[(last)][(-80):4:(-55)]"
            "[(-180):8:(180)]"
        )

        response = requests.get(
            CURRENT_BASE_URL,
            params={"query": query},
            timeout=60
        )

        response.raise_for_status()

        from io import StringIO

        df = pd.read_csv(
            StringIO(response.text),
            skiprows=[1]
        )

        required = {
            "latitude",
            "longitude",
            "u_current",
            "v_current"
        }

        if not required.issubset(df.columns):

            return {
                "available": False,
                "message": (
                    "NOAA current response did not "
                    "contain the expected variables."
                )
            }

        df["u_current"] = pd.to_numeric(
            df["u_current"],
            errors="coerce"
        )

        df["v_current"] = pd.to_numeric(
            df["v_current"],
            errors="coerce"
        )

        df["speed"] = np.sqrt(
            df["u_current"] ** 2
            +
            df["v_current"] ** 2
        )

        df = df[
            df["latitude"] <= -50
        ]

        df = df.dropna()

        return {
            "available": True,
            "data": df
        }

    except Exception as e:

        return {
            "available": False,
            "message": f"NOAA current data unavailable: {e}"
        }


# ============================================================
# LOAD ALL DATA
# ============================================================

with st.spinner("Loading Antarctic datasets..."):

    bed_data = load_bedmachine()
    sea_data = load_sea_ice()
    iceberg_data = load_icebergs()
    current_data = load_currents()


# ============================================================
# DATA STATUS
# ============================================================

dataset_status = {
    "BedMachine": bed_data["available"],
    "Sea Ice": sea_data["available"],
    "Icebergs": iceberg_data["available"],
    "Ocean Currents": current_data["available"],
}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        f"""
        <div style="
            font-size:24px;
            font-weight:800;
            color:{BLUE};
            margin-bottom:2px;
        ">
        🧊 POLARSAFE
        </div>

        <div style="
            font-size:12px;
            color:{MUTED};
            margin-bottom:20px;
        ">
        Antarctic Environmental Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🗺️ MAP LAYERS")

    show_risk = st.checkbox(
        "Environmental Risk",
        True
    )

    show_bathymetry = st.checkbox(
        "Bathymetry",
        False
    )

    show_sea_ice = st.checkbox(
        "Sea Ice Concentration",
        False
    )

    show_currents = st.checkbox(
        "Ocean Currents",
        False
    )

    show_icebergs = st.checkbox(
        "Icebergs",
        True
    )

    st.divider()

    st.markdown("### ⚓ VESSEL")

    vessel = st.selectbox(
        "Vessel profile",
        [
            "Scientific Research Vessel",
            "Standard Cargo Vessel",
            "Polar Class Vessel"
        ]
    )

    vessel_settings = {

        "Scientific Research Vessel": {
            "ice_limit": 55,
            "draft": 6.4
        },

        "Standard Cargo Vessel": {
            "ice_limit": 15,
            "draft": 12.8
        },

        "Polar Class Vessel": {
            "ice_limit": 90,
            "draft": 9.5
        }
    }

    profile = vessel_settings[vessel]

    st.caption(
        f"Maximum preferred ice: "
        f"{profile['ice_limit']}%"
    )

    st.caption(
        f"Typical draft: "
        f"{profile['draft']} m"
    )

    st.divider()

    st.markdown("### ⚙️ RISK WEIGHTS")

    ice_weight = st.slider(
        "Sea ice weight",
        0,
        100,
        35
    )

    current_weight = st.slider(
        "Current weight",
        0,
        100,
        20
    )

    bathy_weight = st.slider(
        "Bathymetry weight",
        0,
        100,
        30
    )

    iceberg_weight = st.slider(
        "Iceberg weight",
        0,
        100,
        15
    )

    total_weight = (
        ice_weight
        + current_weight
        + bathy_weight
        + iceberg_weight
    )

    if total_weight == 0:
        total_weight = 1

    st.caption(
        f"Total weighting: {total_weight}"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    <div class="card"
         style="
         background:
         linear-gradient(
             120deg,
             #0b2d4d,
             #14557c,
             #1e6f96
         );
         color:white;
         border:none;
         padding:26px;
         ">

        <div style="
            font-size:30px;
            font-weight:800;
            color:white;
        ">
            🧊 Antarctica Environmental Risk Explorer
        </div>

        <div style="
            font-size:14px;
            color:#dff3ff;
            margin-top:7px;
        ">
            Bathymetry • Sea Ice • Ocean Currents • Iceberg Monitoring
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA STATUS ROW
# ============================================================

st.markdown("### 📡 Data Status")

status_cols = st.columns(4)

for col, (name, available) in zip(
    status_cols,
    dataset_status.items()
):

    with col:

        if available:

            st.success(
                f"✓ {name} connected"
            )

        else:

            st.warning(
                f"⚠ {name} unavailable"
            )


# ============================================================
# PREPARE GEOSPATIAL DATA
# ============================================================

bathy_grid = None
risk_grid = None
bed_lat = None
bed_lon = None

if bed_data["available"]:

    bed = bed_data["bed"]
    bed_mask = bed_data["mask"]

    bed_lat = bed_data["lat"]
    bed_lon = bed_data["lon"]

    ocean = bed_data["ocean"]

    bathy_grid = np.where(
        ocean,
        bed,
        np.nan
    )


# ============================================================
# HELPER — NEAREST SPATIAL DATA
# ============================================================

def nearest_values(
    source_lat,
    source_lon,
    source_values,
    target_lat,
    target_lon
):

    source_lat = np.asarray(
        source_lat
    ).ravel()

    source_lon = np.asarray(
        source_lon
    ).ravel()

    source_values = np.asarray(
        source_values
    ).ravel()

    valid = (
        np.isfinite(source_lat)
        &
        np.isfinite(source_lon)
        &
        np.isfinite(source_values)
    )

    source_points = np.column_stack(
        [
            source_lat[valid],
            source_lon[valid]
        ]
    )

    source_values = source_values[valid]

    if len(source_points) == 0:

        return np.zeros_like(
            target_lat,
            dtype=float
        )

    tree = cKDTree(
        source_points
    )

    target_points = np.column_stack(
        [
            np.asarray(target_lat).ravel(),
            np.asarray(target_lon).ravel()
        ]
    )

    _, idx = tree.query(
        target_points,
        k=1
    )

    result = source_values[idx]

    return result.reshape(
        np.asarray(target_lat).shape
    )


# ============================================================
# BUILD RISK GRID
# ============================================================

if bed_data["available"]:

    rows, cols = bed.shape

    # Sea ice
    if sea_data["available"]:

        ice_df = sea_data["data"]

        ice_grid = nearest_values(
            ice_df["latitude"].values,
            ice_df["longitude"].values,
            ice_df["ice"].values,
            bed_lat,
            bed_lon
        )

    else:

        ice_grid = np.zeros_like(
            bed,
            dtype=float
        )

    # Current
    if current_data["available"]:

        current_df = current_data["data"]

        current_grid = nearest_values(
            current_df["latitude"].values,
            current_df["longitude"].values,
            current_df["speed"].values,
            bed_lat,
            bed_lon
        )

    else:

        current_grid = np.zeros_like(
            bed,
            dtype=float
        )

    # Bathymetry risk
    depth = np.abs(
        np.minimum(bed, 0)
    )

    depth_risk = 1 - np.clip(
        depth / 6000,
        0,
        1
    )

    # Current risk
    current_risk = np.clip(
        current_grid / 1.5,
        0,
        1
    )

    # Sea ice risk
    ice_risk = np.clip(
        ice_grid / 100,
        0,
        1
    )

    # Iceberg proximity
    iceberg_risk = np.zeros_like(
        bed,
        dtype=float
    )

    if iceberg_data["available"]:

        iceberg_df = iceberg_data["data"]

        points = np.column_stack(
            [
                iceberg_df["latitude"],
                iceberg_df["longitude"]
            ]
        )

        tree = cKDTree(points)

        target = np.column_stack(
            [
                bed_lat.ravel(),
                bed_lon.ravel()
            ]
        )

        distance, _ = tree.query(
            target,
            k=1
        )

        # Approximately 1 degree ≈ 111 km
        iceberg_distance_km = distance * 111

        iceberg_risk = np.exp(
            -iceberg_distance_km / 120
        )

        iceberg_risk = iceberg_risk.reshape(
            bed.shape
        )

    # Weighted composite risk
    risk_grid = (
        ice_risk * ice_weight
        +
        current_risk * current_weight
        +
        depth_risk * bathy_weight
        +
        iceberg_risk * iceberg_weight
    ) / total_weight

    risk_grid *= 100

    risk_grid = np.where(
        ocean,
        risk_grid,
        np.nan
    )


# ============================================================
# TABS
# ============================================================

tab_map, tab_location, tab_seaice, tab_icebergs, tab_route = st.tabs(
    [
        "🗺️ Risk Map",
        "📍 Location Assessment",
        "🧊 Sea Ice",
        "🚨 Icebergs",
        "🧭 Route Planner"
    ]
)


# ============================================================
# TAB 1 — MAP
# ============================================================

with tab_map:

    st.markdown(
        '<div class="section-title">'
        'Environmental Risk Map'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Integrated view of Antarctic environmental conditions.'
        '</div>',
        unsafe_allow_html=True
    )

    if not bed_data["available"]:

        st.warning(
            "BedMachine is not available in this deployment. "
            "Add the BedMachine-derived grid to the repository to "
            "activate the full risk map."
        )

    fig = go.Figure()

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    if (
        show_risk
        and risk_grid is not None
    ):

        # Reduce display density
        stride = max(
            1,
            risk_grid.shape[0] // 90
        )

        r = risk_grid[
            ::stride,
            ::stride
        ]

        la = bed_lat[
            ::stride,
            ::stride
        ]

        lo = bed_lon[
            ::stride,
            ::stride
        ]

        fig.add_trace(
            go.Scattergeo(
                lat=la.ravel(),
                lon=lo.ravel(),
                mode="markers",
                marker=dict(
                    size=5,
                    color=r.ravel(),
                    colorscale=[
                        [0.00, "#16a34a"],
                        [0.35, "#84cc16"],
                        [0.55, "#facc15"],
                        [0.75, "#fb923c"],
                        [1.00, "#dc2626"]
                    ],
                    cmin=0,
                    cmax=100,
                    opacity=0.78,
                    colorbar=dict(
                        title="Risk",
                        ticksuffix=""
                    )
                ),
                name="Environmental Risk",
                customdata=np.column_stack(
                    [
                        la.ravel(),
                        lo.ravel(),
                        r.ravel()
                    ]
                ),
                hovertemplate=(
                    "<b>Environmental Risk</b><br>"
                    "Latitude: %{customdata[0]:.2f}°<br>"
                    "Longitude: %{customdata[1]:.2f}°<br>"
                    "Risk: %{customdata[2]:.1f}/100"
                    "<extra></extra>"
                )
            )
        )

    # --------------------------------------------------------
    # Bathymetry
    # --------------------------------------------------------

    if (
        show_bathymetry
        and bathy_grid is not None
    ):

        stride = max(
            1,
            bathy_grid.shape[0] // 80
        )

        z = bathy_grid[
            ::stride,
            ::stride
        ]

        la = bed_lat[
            ::stride,
            ::stride
        ]

        lo = bed_lon[
            ::stride,
            ::stride
        ]

        fig.add_trace(
            go.Scattergeo(
                lat=la.ravel(),
                lon=lo.ravel(),
                mode="markers",
                marker=dict(
                    size=4,
                    color=z.ravel(),
                    colorscale="Blues",
                    reversescale=False,
                    opacity=0.65,
                    colorbar=dict(
                        title="Bed (m)"
                    )
                ),
                name="Bathymetry",
                hovertemplate=(
                    "Bed depth: %{marker.color:.0f} m"
                    "<extra></extra>"
                )
            )
        )

    # --------------------------------------------------------
    # Sea Ice
    # --------------------------------------------------------

    if (
        show_sea_ice
        and sea_data["available"]
    ):

        df = sea_data["data"]

        fig.add_trace(
            go.Scattergeo(
                lat=df["latitude"],
                lon=df["longitude"],
                mode="markers",
                marker=dict(
                    size=4,
                    color=df["ice"],
                    colorscale="Blues",
                    cmin=0,
                    cmax=100,
                    opacity=0.65,
                    colorbar=dict(
                        title="Ice %"
                    )
                ),
                name="Sea Ice",
                hovertemplate=(
                    "Sea ice: %{marker.color:.0f}%"
                    "<br>Lat: %{lat:.2f}°"
                    "<br>Lon: %{lon:.2f}°"
                    "<extra></extra>"
                )
            )
        )

    # --------------------------------------------------------
    # Currents
    # --------------------------------------------------------

    if (
        show_currents
        and current_data["available"]
    ):

        df = current_data["data"]

        # Keep a manageable number of arrows
        if len(df) > 500:
            df = df.sample(
                500,
                random_state=42
            )

        for _, row in df.iterrows():

            scale = 1.2

            fig.add_trace(
                go.Scattergeo(
                    lat=[
                        row["latitude"],
                        row["latitude"]
                        + row["v_current"] * scale
                    ],
                    lon=[
                        row["longitude"],
                        row["longitude"]
                        + row["u_current"] * scale
                    ],
                    mode="lines",
                    line=dict(
                        color=CYAN,
                        width=1.5
                    ),
                    showlegend=False,
                    hoverinfo="skip"
                )
            )

    # --------------------------------------------------------
    # Icebergs
    # --------------------------------------------------------

    if (
        show_icebergs
        and iceberg_data["available"]
    ):

        df = iceberg_data["data"]

        fig.add_trace(
            go.Scattergeo(
                lat=df["latitude"],
                lon=df["longitude"],
                mode="markers+text",
                text=df["iceberg"],
                textposition="top center",
                marker=dict(
                    size=10,
                    color=RED,
                    symbol="diamond",
                    line=dict(
                        color=WHITE,
                        width=1
                    )
                ),
                name="Icebergs",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Latitude: %{lat:.2f}°<br>"
                    "Longitude: %{lon:.2f}°"
                    "<extra></extra>"
                )
            )
        )

    # --------------------------------------------------------
    # Map
    # --------------------------------------------------------

    fig.update_geos(
        projection_type="stereographic",
        center=dict(
            lat=-90,
            lon=0
        ),
        projection_scale=1.35,
        showland=True,
        landcolor="#dbeaf3",
        showocean=True,
        oceancolor="#eaf6ff",
        showcoastlines=True,
        coastlinecolor="#315b73",
        coastlinewidth=1,
        showcountries=False,
        showlakes=False,
        lataxis=dict(
            range=[-90, -48]
        )
    )

    fig.update_layout(
        height=650,
        margin=dict(
            l=5,
            r=5,
            t=10,
            b=5
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color=TEXT
        ),
        legend=dict(
            orientation="h",
            y=0.02,
            x=0.02,
            bgcolor="rgba(255,255,255,0.85)"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="environmental_risk_map",
        config={
            "displayModeBar": False,
            "responsive": True
        }
    )


# ============================================================
# TAB 2 — LOCATION ASSESSMENT
# ============================================================

with tab_location:

    st.markdown(
        '<div class="section-title">'
        '📍 Location Assessment'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Select a location to inspect the environmental conditions '
        'at the nearest available data cell.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=-48.0,
            value=-67.568,
            step=0.01
        )

    with col2:

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-68.129,
            step=0.01
        )

    assess = st.button(
        "🔍 Assess Location",
        use_container_width=True
    )

    if assess:

        if not bed_data["available"]:

            st.warning(
                "BedMachine is required for the full "
                "location risk assessment."
            )

        else:

            distance = (
                (bed_lat - latitude) ** 2
                +
                (bed_lon - longitude) ** 2
            )

            distance[
                ~np.isfinite(distance)
            ] = np.inf

            idx = np.unravel_index(
                np.argmin(distance),
                distance.shape
            )

            row, col = idx

            nearest_lat = bed_lat[
                row,
                col
            ]

            nearest_lon = bed_lon[
                row,
                col
            ]

            score = risk_grid[
                row,
                col
            ]

            if not np.isfinite(score):
                score = 0

            bed_value = bed[
                row,
                col
            ]

            slope_value = bed_data[
                "slope"
            ][
                row,
                col
            ]

            # Get local sea ice
            local_ice = 0

            if sea_data["available"]:

                local_ice = nearest_values(
                    sea_data["data"]["latitude"],
                    sea_data["data"]["longitude"],
                    sea_data["data"]["ice"],
                    np.array([[latitude]]),
                    np.array([[longitude]])
                )[0, 0]

            # Get current
            local_current = 0

            if current_data["available"]:

                local_current = nearest_values(
                    current_data["data"]["latitude"],
                    current_data["data"]["longitude"],
                    current_data["data"]["speed"],
                    np.array([[latitude]]),
                    np.array([[longitude]])
                )[0, 0]

            # Find nearby iceberg
            iceberg_distance = None
            nearest_berg = None

            if iceberg_data["available"]:

                iceberg_df = iceberg_data["data"]

                distances = (
                    (
                        iceberg_df["latitude"]
                        - latitude
                    ) ** 2
                    +
                    (
                        iceberg_df["longitude"]
                        - longitude
                    ) ** 2
                )

                nearest_idx = distances.idxmin()

                nearest_berg = iceberg_df.loc[
                    nearest_idx,
                    "iceberg"
                ]

                iceberg_distance = math.sqrt(
                    distances.loc[
                        nearest_idx
                    ]
                ) * 111

            st.divider()

            if score >= 70:

                st.markdown(
                    f"""
                    <div class="risk-high">
                        <h2>🔴 HIGH ENVIRONMENTAL RISK</h2>
                        <h1>{score:.0f}/100</h1>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif score >= 40:

                st.markdown(
                    f"""
                    <div class="risk-medium">
                        <h2>🟠 MODERATE ENVIRONMENTAL RISK</h2>
                        <h1>{score:.0f}/100</h1>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="risk-low">
                        <h2>🟢 LOWER ENVIRONMENTAL RISK</h2>
                        <h1>{score:.0f}/100</h1>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                f"### Nearest data cell: "
                f"{nearest_lat:.3f}°, "
                f"{nearest_lon:.3f}°"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Sea Ice",
                    f"{local_ice:.1f}%"
                )

            with c2:

                st.metric(
                    "Current",
                    f"{local_current:.2f} m/s"
                )

            with c3:

                st.metric(
                    "Bed",
                    f"{bed_value:.0f} m"
                )

            with c4:

                st.metric(
                    "Bed Slope",
                    f"{slope_value:.3f}"
                )

            st.markdown("### Why this score?")

            reasons = []

            if local_ice >= 70:
                reasons.append(
                    "High sea-ice concentration"
                )

            elif local_ice >= 40:
                reasons.append(
                    "Moderate sea-ice concentration"
                )

            if local_current >= 1:
                reasons.append(
                    "Strong ocean current"
                )

            elif local_current >= 0.5:
                reasons.append(
                    "Moderate ocean current"
                )

            if bed_value > -200:
                reasons.append(
                    "Relatively shallow seabed"
                )

            if iceberg_distance is not None:

                if iceberg_distance < 100:

                    reasons.append(
                        f"Near iceberg {nearest_berg}"
                    )

                elif iceberg_distance < 250:

                    reasons.append(
                        f"Iceberg activity within ~"
                        f"{iceberg_distance:.0f} km"
                    )

            if not reasons:

                reasons.append(
                    "No major environmental indicator "
                    "was detected at this location."
                )

            for reason in reasons:

                st.info(
                    f"• {reason}"
                )


# ============================================================
# TAB 3 — SEA ICE ANALYSIS
# ============================================================

with tab_seaice:

    st.markdown(
        '<div class="section-title">'
        '🧊 Sea Ice Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Antarctic sea-ice concentration from the available dataset.'
        '</div>',
        unsafe_allow_html=True
    )

    if not sea_data["available"]:

        st.error(
            sea_data["message"]
        )

    else:

        ice_df = sea_data["data"]

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Mean concentration",
                f"{ice_df['ice'].mean():.1f}%"
            )

        with c2:

            st.metric(
                "Maximum",
                f"{ice_df['ice'].max():.1f}%"
            )

        with c3:

            st.metric(
                "Data points",
                f"{len(ice_df):,}"
            )

        fig = go.Figure()

        fig.add_trace(
            go.Scattergeo(
                lat=ice_df["latitude"],
                lon=ice_df["longitude"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=ice_df["ice"],
                    colorscale="Blues",
                    cmin=0,
                    cmax=100,
                    opacity=0.75,
                    colorbar=dict(
                        title="Ice %"
                    )
                ),
                hovertemplate=(
                    "<b>Sea Ice Concentration</b><br>"
                    "%{marker.color:.1f}%<br>"
                    "Lat: %{lat:.2f}°<br>"
                    "Lon: %{lon:.2f}°"
                    "<extra></extra>"
                )
            )
        )

        fig.update_geos(
            projection_type="stereographic",
            center=dict(
                lat=-90,
                lon=0
            ),
            projection_scale=1.35,
            showland=True,
            landcolor="#dbeaf3",
            showocean=True,
            oceancolor="#f3fbff",
            showcoastlines=True,
            coastlinecolor="#315b73"
        )

        fig.update_layout(
            height=600,
            margin=dict(
                l=0,
                r=0,
                t=0,
                b=0
            ),
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="sea_ice_analysis_map"
        )

        st.info(
            f"Dataset observation: {sea_data['date']}"
        )


# ============================================================
# TAB 4 — ICEBERG MONITOR
# ============================================================

with tab_icebergs:

    st.markdown(
        '<div class="section-title">'
        '🚨 Antarctic Iceberg Monitor'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Current large iceberg positions derived from satellite '
        'scatterometer observations.'
        '</div>',
        unsafe_allow_html=True
    )

    if not iceberg_data["available"]:

        st.error(
            iceberg_data["message"]
        )

    else:

        bergs = iceberg_data["data"]

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Tracked icebergs",
                len(bergs)
            )

        with c2:

            st.metric(
                "Southernmost",
                f"{bergs['latitude'].min():.1f}°S"
            )

        with c3:

            st.metric(
                "Latest source",
                "NASA SCP"
            )

        fig = go.Figure()

        fig.add_trace(
            go.Scattergeo(
                lat=bergs["latitude"],
                lon=bergs["longitude"],
                mode="markers+text",
                text=bergs["iceberg"],
                textposition="top center",
                marker=dict(
                    size=11,
                    color=RED,
                    symbol="diamond",
                    line=dict(
                        color=WHITE,
                        width=1
                    )
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Latitude: %{lat:.2f}°<br>"
                    "Longitude: %{lon:.2f}°"
                    "<extra></extra>"
                ),
                name="Iceberg"
            )
        )

        fig.update_geos(
            projection_type="stereographic",
            center=dict(
                lat=-90,
                lon=0
            ),
            projection_scale=1.35,
            showland=True,
            landcolor="#dbeaf3",
            showocean=True,
            oceancolor="#f3fbff",
            showcoastlines=True,
            coastlinecolor="#315b73"
        )

        fig.update_layout(
            height=580,
            margin=dict(
                l=0,
                r=0,
                t=0,
                b=0
            ),
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="iceberg_monitor_map"
        )

        st.markdown("### Current Iceberg Positions")

        display_df = bergs.copy()

        display_df.columns = [
            "Iceberg",
            "Latitude",
            "Longitude",
            "Observation"
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Iceberg tracking is supplementary environmental intelligence "
            "and should not replace official navigation products."
        )


# ============================================================
# ROUTE HELPERS
# ============================================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = lat2 - lat1

    dlon = np.radians(
        lon2 - lon1
    )

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(dlon / 2) ** 2
    )

    return (
        6371
        *
        2
        *
        np.arcsin(
            np.sqrt(a)
        )
    )


def nearest_ocean_cell(
    lat,
    lon,
    grid_lat,
    grid_lon,
    ocean
):

    distance = (
        (grid_lat - lat) ** 2
        +
        (grid_lon - lon) ** 2
    )

    distance[
        ~ocean
    ] = np.inf

    if not np.isfinite(
        distance
    ).any():

        return None

    return np.unravel_index(
        np.argmin(distance),
        distance.shape
    )


# ============================================================
# A* ROUTER
# ============================================================

def calculate_route(
    risk,
    ocean,
    start,
    goal,
    ice_limit,
    bed,
    current_grid,
    ice_grid
):

    rows, cols = risk.shape

    blocked = ~ocean

    # Ice restriction
    blocked |= (
        ice_grid > ice_limit
    )

    # Very shallow areas
    blocked |= (
        bed > -50
    )

    sr, sc = start
    gr, gc = goal

    blocked[sr, sc] = False
    blocked[gr, gc] = False

    # Make route cost
    cost = (
        1.0
        +
        np.nan_to_num(
            risk,
            nan=100
        ) / 100.0 * 8
    )

    cost = np.where(
        blocked,
        np.inf,
        cost
    )

    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 1.414),
        (-1, 1, 1.414),
        (1, -1, 1.414),
        (1, 1, 1.414)
    ]

    import heapq

    def heuristic(a, b):

        return math.hypot(
            a[0] - b[0],
            a[1] - b[1]
        )

    open_set = []

    heapq.heappush(
        open_set,
        (
            heuristic(start, goal),
            0,
            start
        )
    )

    came_from = {}

    g_score = {
        start: 0
    }

    visited = set()

    while open_set:

        _, current_cost, current = (
            heapq.heappop(open_set)
        )

        if current in visited:
            continue

        visited.add(current)

        if current == goal:

            path = [
                current
            ]

            while current in came_from:

                current = came_from[
                    current
                ]

                path.append(
                    current
                )

            path.reverse()

            return path

        r, c = current

        for dr, dc, step in neighbors:

            nr = r + dr
            nc = c + dc

            if (
                nr < 0
                or nr >= rows
                or nc < 0
                or nc >= cols
            ):
                continue

            if not np.isfinite(
                cost[nr, nc]
            ):
                continue

            new_cost = (
                current_cost
                +
                cost[nr, nc] * step
            )

            neighbour = (
                nr,
                nc
            )

            if new_cost < g_score.get(
                neighbour,
                np.inf
            ):

                g_score[
                    neighbour
                ] = new_cost

                came_from[
                    neighbour
                ] = current

                f = (
                    new_cost
                    +
                    heuristic(
                        neighbour,
                        goal
                    )
                )

                heapq.heappush(
                    open_set,
                    (
                        f,
                        new_cost,
                        neighbour
                    )
                )

    return None


# ============================================================
# TAB 5 — ROUTE PLANNER
# ============================================================

with tab_route:

    st.markdown(
        '<div class="section-title">'
        '🧭 Safer Antarctic Route Planner'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Find a route that balances distance with environmental risk.'
        '</div>',
        unsafe_allow_html=True
    )

    if not bed_data["available"]:

        st.warning(
            "The route planner requires the BedMachine dataset "
            "or a derived BedMachine grid in the repository."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### 🟢 Departure")

            start_name = st.selectbox(
                "Source",
                list(
                    ANTARCTIC_LOCATIONS.keys()
                ),
                index=0
            )

            start_lat, start_lon = (
                ANTARCTIC_LOCATIONS[
                    start_name
                ]
            )

        with col2:

            st.markdown("### 🔵 Destination")

            destination_options = [
                x for x in
                ANTARCTIC_LOCATIONS.keys()
                if x != start_name
            ]

            end_name = st.selectbox(
                "Destination",
                destination_options,
                index=0
            )

            end_lat, end_lon = (
                ANTARCTIC_LOCATIONS[
                    end_name
                ]
            )

        st.markdown(
            f"""
            <div class="card">
            <b>Route:</b>
            {start_name}
            →
            {end_name}

            <br><br>

            <b>Departure:</b>
            {start_lat:.3f}°, {start_lon:.3f}°

            <br>

            <b>Destination:</b>
            {end_lat:.3f}°, {end_lon:.3f}°
            </div>
            """,
            unsafe_allow_html=True
        )

        route_button = st.button(
            "🧭 CALCULATE SAFER ROUTE",
            use_container_width=True
        )

        if route_button:

            start = nearest_ocean_cell(
                start_lat,
                start_lon,
                bed_lat,
                bed_lon,
                bed_data["ocean"]
            )

            goal = nearest_ocean_cell(
                end_lat,
                end_lon,
                bed_lat,
                bed_lon,
                bed_data["ocean"]
            )

            if (
                start is None
                or goal is None
            ):

                st.error(
                    "Could not find navigable ocean cells "
                    "near the selected locations."
                )

            else:

                # ------------------------------------------------
                # Rebuild route-specific risk grid
                # ------------------------------------------------

                route_risk = np.nan_to_num(
                    risk_grid,
                    nan=100
                )

                path = calculate_route(
                    route_risk,
                    bed_data["ocean"],
                    start,
                    goal,
                    profile["ice_limit"],
                    bed,
                    current_grid,
                    ice_grid
                )

                if path is None:

                    st.error(
                        "No viable route was found under the "
                        "current environmental constraints."
                    )

                    st.info(
                        "Try selecting a Polar Class Vessel "
                        "or reduce the environmental constraints."
                    )

                else:

                    path = np.array(
                        path
                    )

                    path_lat = bed_lat[
                        path[:, 0],
                        path[:, 1]
                    ]

                    path_lon = bed_lon[
                        path[:, 0],
                        path[:, 1]
                    ]

                    route_risk_values = risk_grid[
                        path[:, 0],
                        path[:, 1]
                    ]

                    route_ice_values = ice_grid[
                        path[:, 0],
                        path[:, 1]
                    ]

                    route_current_values = current_grid[
                        path[:, 0],
                        path[:, 1]
                    ]

                    route_depth_values = bed[
                        path[:, 0],
                        path[:, 1]
                    ]

                    # ------------------------------------------------
                    # Distance
                    # ------------------------------------------------

                    distance_segments = haversine_km(
                        path_lat[:-1],
                        path_lon[:-1],
                        path_lat[1:],
                        path_lon[1:]
                    )

                    total_km = float(
                        np.nansum(
                            distance_segments
                        )
                    )

                    total_nm = (
                        total_km
                        * 0.539957
                    )

                    direct_km = float(
                        haversine_km(
                            start_lat,
                            start_lon,
                            end_lat,
                            end_lon
                        )
                    )

                    direct_nm = (
                        direct_km
                        * 0.539957
                    )

                    # ------------------------------------------------
                    # Environmental metrics
                    # ------------------------------------------------

                    max_risk = float(
                        np.nanmax(
                            route_risk_values
                        )
                    )

                    avg_risk = float(
                        np.nanmean(
                            route_risk_values
                        )
                    )

                    max_ice = float(
                        np.nanmax(
                            route_ice_values
                        )
                    )

                    avg_current = float(
                        np.nanmean(
                            route_current_values
                        )
                    )

                    min_depth = float(
                        np.nanmin(
                            route_depth_values
                        )
                    )

                    # ------------------------------------------------
                    # ETA
                    # ------------------------------------------------

                    speed = {

                        "Scientific Research Vessel": 12,

                        "Standard Cargo Vessel": 16,

                        "Polar Class Vessel": 14

                    }[vessel]

                    speed_factor = max(
                        0.5,
                        1 -
                        (
                            max_ice
                            / 100
                            * 0.4
                        )
                    )

                    effective_speed = (
                        speed
                        * speed_factor
                    )

                    hours = (
                        total_nm
                        /
                        max(
                            effective_speed,
                            1
                        )
                    )

                    # ------------------------------------------------
                    # Risk classification
                    # ------------------------------------------------

                    if avg_risk >= 70:
                        risk_level = "HIGH"
                    elif avg_risk >= 40:
                        risk_level = "MODERATE"
                    else:
                        risk_level = "LOW"

                    # ------------------------------------------------
                    # Route map
                    # ------------------------------------------------

                    fig_route = go.Figure()

                    if show_risk:

                        stride = max(
                            1,
                            risk_grid.shape[0] // 70
                        )

                        r = risk_grid[
                            ::stride,
                            ::stride
                        ]

                        la = bed_lat[
                            ::stride,
                            ::stride
                        ]

                        lo = bed_lon[
                            ::stride,
                            ::stride
                        ]

                        fig_route.add_trace(
                            go.Scattergeo(
                                lat=la.ravel(),
                                lon=lo.ravel(),
                                mode="markers",
                                marker=dict(
                                    size=4,
                                    color=r.ravel(),
                                    colorscale=[
                                        [0, GREEN],
                                        [0.5, YELLOW],
                                        [1, RED]
                                    ],
                                    cmin=0,
                                    cmax=100,
                                    opacity=0.35,
                                    showscale=True,
                                    colorbar=dict(
                                        title="Risk"
                                    )
                                ),
                                name="Risk"
                            )
                        )

                    fig_route.add_trace(
                        go.Scattergeo(
                            lat=path_lat,
                            lon=path_lon,
                            mode="lines",
                            line=dict(
                                color="#0b2d4d",
                                width=5
                            ),
                            name="Safer Route"
                        )
                    )

                    fig_route.add_trace(
                        go.Scattergeo(
                            lat=[start_lat],
                            lon=[start_lon],
                            mode="markers+text",
                            text=["START"],
                            textposition="top center",
                            marker=dict(
                                size=14,
                                color=GREEN,
                                symbol="circle"
                            ),
                            name="Departure"
                        )
                    )

                    fig_route.add_trace(
                        go.Scattergeo(
                            lat=[end_lat],
                            lon=[end_lon],
                            mode="markers+text",
                            text=["DESTINATION"],
                            textposition="top center",
                            marker=dict(
                                size=14,
                                color="#2563eb",
                                symbol="star"
                            ),
                            name="Destination"
                        )
                    )

                    # Add icebergs
                    if iceberg_data["available"]:

                        bergs = iceberg_data["data"]

                        fig_route.add_trace(
                            go.Scattergeo(
                                lat=bergs["latitude"],
                                lon=bergs["longitude"],
                                mode="markers",
                                marker=dict(
                                    size=8,
                                    color=RED,
                                    symbol="diamond"
                                ),
                                name="Icebergs",
                                hovertemplate=(
                                    "<b>%{text}</b>"
                                    "<br>%{lat:.2f}°"
                                    "<br>%{lon:.2f}°"
                                    "<extra></extra>"
                                ),
                                text=bergs["iceberg"]
                            )
                        )

                    fig_route.update_geos(
                        projection_type="stereographic",
                        center=dict(
                            lat=-90,
                            lon=0
                        ),
                        projection_scale=1.35,
                        showland=True,
                        landcolor="#dbeaf3",
                        showocean=True,
                        oceancolor="#f3fbff",
                        showcoastlines=True,
                        coastlinecolor="#315b73"
                    )

                    fig_route.update_layout(
                        height=620,
                        margin=dict(
                            l=0,
                            r=0,
                            t=0,
                            b=0
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(
                            color=TEXT
                        ),
                        legend=dict(
                            orientation="h",
                            bgcolor="rgba(255,255,255,0.85)"
                        )
                    )

                    st.plotly_chart(
                        fig_route,
                        use_container_width=True,
                        key="route_map"
                    )

                    # ------------------------------------------------
                    # Route KPIs
                    # ------------------------------------------------

                    st.markdown("### 📊 Route Summary")

                    c1, c2, c3, c4 = st.columns(4)

                    with c1:

                        st.metric(
                            "Distance",
                            f"{total_nm:,.0f} NM"
                        )

                    with c2:

                        st.metric(
                            "Estimated time",
                            f"{hours:.1f} hrs"
                        )

                    with c3:

                        st.metric(
                            "Average risk",
                            f"{avg_risk:.0f}/100"
                        )

                    with c4:

                        st.metric(
                            "Maximum ice",
                            f"{max_ice:.0f}%"
                        )

                    if risk_level == "HIGH":

                        st.error(
                            "🔴 HIGH RISK ROUTE — "
                            "Review environmental conditions "
                            "before transit."
                        )

                    elif risk_level == "MODERATE":

                        st.warning(
                            "🟠 MODERATE RISK ROUTE — "
                            "Environmental conditions require monitoring."
                        )

                    else:

                        st.success(
                            "🟢 LOWER RISK ROUTE — "
                            "No major environmental constraint "
                            "was encountered on the calculated path."
                        )

                    # ------------------------------------------------
                    # Route environmental profile
                    # ------------------------------------------------

                    st.markdown(
                        "### 🌊 Environmental Profile"
                    )

                    c1, c2, c3, c4 = st.columns(4)

                    with c1:

                        st.metric(
                            "Peak risk",
                            f"{max_risk:.0f}/100"
                        )

                    with c2:

                        st.metric(
                            "Avg current",
                            f"{avg_current:.2f} m/s"
                        )

                    with c3:

                        st.metric(
                            "Deepest bed",
                            f"{min_depth:.0f} m"
                        )

                    with c4:

                        detour = (
                            (
                                total_km
                                /
                                max(
                                    direct_km,
                                    1
                                )
                            )
                            - 1
                        ) * 100

                        st.metric(
                            "Route detour",
                            f"{detour:.1f}%"
                        )

                    # ------------------------------------------------
                    # Depth chart
                    # ------------------------------------------------

                    chart_col1, chart_col2 = st.columns(2)

                    with chart_col1:

                        fig_depth = go.Figure()

                        fig_depth.add_trace(
                            go.Scatter(
                                x=np.arange(
                                    len(
                                        route_depth_values
                                    )
                                ),
                                y=route_depth_values,
                                mode="lines",
                                fill="tozeroy",
                                line=dict(
                                    color=BLUE_3,
                                    width=2
                                ),
                                name="Bed depth"
                            )
                        )

                        fig_depth.update_layout(
                            title="Bathymetry Along Route",
                            height=350,
                            xaxis_title="Waypoint",
                            yaxis_title="Bed elevation (m)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(255,255,255,0.7)",
                            font=dict(
                                color=TEXT
                            )
                        )

                        st.plotly_chart(
                            fig_depth,
                            use_container_width=True,
                            key="route_depth_chart"
                        )

                    with chart_col2:

                        fig_env = go.Figure()

                        fig_env.add_trace(
                            go.Scatter(
                                x=np.arange(
                                    len(
                                        route_ice_values
                                    )
                                ),
                                y=route_ice_values,
                                mode="lines",
                                name="Sea Ice"
                            )
                        )

                        fig_env.add_trace(
                            go.Scatter(
                                x=np.arange(
                                    len(
                                        route_current_values
                                    )
                                ),
                                y=route_current_values * 50,
                                mode="lines",
                                name="Current × 50"
                            )
                        )

                        fig_env.update_layout(
                            title="Environmental Conditions Along Route",
                            height=350,
                            xaxis_title="Waypoint",
                            yaxis_title="Value",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(255,255,255,0.7)",
                            font=dict(
                                color=TEXT
                            )
                        )

                        st.plotly_chart(
                            fig_env,
                            use_container_width=True,
                            key="route_environment_chart"
                        )

                    # ------------------------------------------------
                    # Route report
                    # ------------------------------------------------

                    report = pd.DataFrame({
                        "latitude": path_lat,
                        "longitude": path_lon,
                        "risk_score": route_risk_values,
                        "sea_ice_percent": route_ice_values,
                        "current_speed_ms": route_current_values,
                        "bed_depth_m": route_depth_values
                    })

                    st.download_button(
                        "⬇️ Download Route Assessment",
                        data=report.to_csv(
                            index=False
                        ),
                        file_name="polarsafe_route_assessment.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    f"""
    <div style="
        text-align:center;
        padding:15px;
        color:{MUTED};
        font-size:12px;
    ">

    <b style="color:{BLUE};">
    POLARSAFE
    </b>

    · Antarctic Environmental Risk Explorer

    <br>

    BedMachine Antarctica · NOAA CoastWatch · NASA SCP

    </div>
    """,
    unsafe_allow_html=True
)
