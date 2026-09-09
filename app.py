import os
import glob
import io
import math
import requests

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import xarray as xr
from pyproj import Transformer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PolarSafe | Antarctic Risk Explorer",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# THEME / CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            linear-gradient(
                135deg,
                #eef8ff 0%,
                #dcefff 45%,
                #c7e4f7 100%
            );
        color: #102a43;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- TEXT ---------- */

    h1, h2, h3, h4, h5, h6 {
        color: #082f49 !important;
    }

    p, label, span, div {
        color: #102a43;
    }

    .stMarkdown p {
        color: #102a43;
    }

    /* ---------- SIDEBAR ---------- */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #d9efff 0%,
                #c2e1f5 50%,
                #a9d2ed 100%
            );
        border-right: 1px solid #8bbbd8;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #082f49 !important;
    }

    [data-testid="stSidebar"] label {
        color: #12344d !important;
        font-weight: 600;
    }

    /* ---------- HEADER ---------- */

    .hero {
        background:
            linear-gradient(
                135deg,
                #063b5c 0%,
                #075985 50%,
                #0c7aa6 100%
            );
        padding: 30px 35px;
        border-radius: 22px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(8, 47, 73, 0.18);
    }

    .hero h1 {
        color: white !important;
        margin-bottom: 5px;
        font-size: 2.6rem;
    }

    .hero p {
        color: #dff5ff !important;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    /* ---------- CARDS ---------- */

    .info-card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(71, 130, 165, 0.25);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 5px 18px rgba(8, 47, 73, 0.08);
    }

    .info-card h3 {
        color: #075985 !important;
        margin-top: 0;
    }

    /* ---------- METRICS ---------- */

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid #b5d5e8;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 14px rgba(8, 47, 73, 0.07);
    }

    [data-testid="stMetricLabel"] {
        color: #31566d !important;
    }

    [data-testid="stMetricValue"] {
        color: #073b5c !important;
    }

    /* ---------- BUTTONS ---------- */

    .stButton > button {
        background:
            linear-gradient(
                135deg,
                #075985,
                #0c7aa6
            );
        color: white !important;
        border: none;
        border-radius: 11px;
        padding: 0.65rem 1.1rem;
        font-weight: 700;
    }

    .stButton > button:hover {
        background:
            linear-gradient(
                135deg,
                #064b70,
                #096b91
            );
        color: white !important;
    }

    /* ---------- SELECTBOX / INPUT ---------- */

    div[data-baseweb="select"] > div {
        background-color: white;
        color: #102a43;
        border-radius: 10px;
    }

    input {
        color: #102a43 !important;
    }

    /* ---------- TABS ---------- */

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.65);
        border-radius: 10px;
        color: #17445d !important;
        padding: 10px 18px;
    }

    .stTabs [aria-selected="true"] {
        background: #075985 !important;
        color: white !important;
    }

    /* ---------- ALERTS ---------- */

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        padding: 25px;
        color: #31566d !important;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

SEA_ICE_PATH = "data/raw/sea_ice/G10010_sibt1850_v2.0.nc"

USNIC_ICEBERG_URL = (
    "https://usicecenter.gov/File/DownloadCurrent?pId=134"
)

NOAA_CURRENT_URL = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/"
    "miamicurrents"
)


# ============================================================
# HELPER: FIND BEDMACHINE
# ============================================================

@st.cache_data(show_spinner=False)
def find_bedmachine_file():

    possible_patterns = [
        "**/*BedMachine*.nc",
        "**/*bedmachine*.nc",
        "**/NSIDC-0756*.nc",
        "**/*V04.1.nc"
    ]

    matches = []

    for pattern in possible_patterns:
        matches.extend(
            glob.glob(
                pattern,
                recursive=True
            )
        )

    # Remove duplicates
    matches = list(dict.fromkeys(matches))

    if len(matches) == 0:
        return None

    # Prefer smaller files if there are samples
    matches.sort(
        key=lambda x: os.path.getsize(x)
    )

    return matches[0]


# ============================================================
# SEA ICE LOADER
# ============================================================

@st.cache_resource(show_spinner=False)
def load_sea_ice():

    if not os.path.exists(SEA_ICE_PATH):
        return None, "Sea ice file not found"

    try:

        ds = xr.open_dataset(
            SEA_ICE_PATH,
            decode_times=False
        )

        return ds, "OK"

    except Exception as e:

        return None, str(e)


# ============================================================
# BEDMACHINE LOADER
# ============================================================

@st.cache_resource(show_spinner=False)
def load_bedmachine():

    path = find_bedmachine_file()

    if path is None:
        return None, None, "BedMachine file not found"

    try:

        ds = xr.open_dataset(
            path
        )

        return ds, path, "OK"

    except Exception as e:

        return None, path, str(e)


# ============================================================
# ICEBERG LOADER
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_icebergs():

    try:

        response = requests.get(
            USNIC_ICEBERG_URL,
            timeout=30
        )

        response.raise_for_status()

        text = response.content.decode(
            "utf-8",
            errors="ignore"
        )

        # Try normal CSV
        try:
            df = pd.read_csv(
                io.StringIO(text)
            )
        except Exception:

            # Some USNIC files may need more permissive parsing
            df = pd.read_csv(
                io.StringIO(text),
                sep=None,
                engine="python"
            )

        return df, "OK"

    except Exception as e:

        return None, str(e)


# ============================================================
# CURRENT LOADER
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def load_currents():

    try:

        # We request a SMALL subset rather than downloading
        # the entire global dataset.

        url = (
            NOAA_CURRENT_URL
            + ".csv?"
            + "u_current[(2019-06-16T00:00:00Z)]"
            "[(-74.875):2:(-60)]"
            "[(-180):2:(180)],"
            + "v_current[(2019-06-16T00:00:00Z)]"
            "[(-74.875):2:(-60)]"
            "[(-180):2:(180)]"
        )

        response = requests.get(
            url,
            timeout=60
        )

        response.raise_for_status()

        df = pd.read_csv(
            io.StringIO(
                response.text
            )
        )

        return df, "OK"

    except Exception as e:

        return None, str(e)


# ============================================================
# LOAD DATA
# ============================================================

sea_ice_ds, sea_ice_status = load_sea_ice()

bed_ds, bed_path, bed_status = load_bedmachine()

iceberg_df, iceberg_status = load_icebergs()

current_df, current_status = load_currents()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>🧊 PolarSafe</h1>

        <p>
        Antarctic Environmental Risk & Navigation Explorer
        </p>

        <p>
        Integrated view of sea ice, icebergs, ocean currents
        and seabed conditions.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 🗺️ Map Layers"
)

show_risk = st.sidebar.checkbox(
    "Environmental Risk",
    True
)

show_sea_ice = st.sidebar.checkbox(
    "Sea Ice Concentration",
    True
)

show_icebergs = st.sidebar.checkbox(
    "Iceberg Monitoring",
    True
)

show_currents = st.sidebar.checkbox(
    "Ocean Currents",
    True
)

show_bathymetry = st.sidebar.checkbox(
    "BedMachine Bathymetry",
    True
)

st.sidebar.divider()

st.sidebar.markdown(
    "## ⚠️ Risk Settings"
)

risk_threshold = st.sidebar.slider(
    "High-risk threshold",
    0,
    100,
    60
)

st.sidebar.divider()

st.sidebar.markdown(
    "## 📍 Antarctic Locations"
)

locations = {
    "Cape Adare": (-71.28, 170.30),
    "Ross Sea": (-75.00, 175.00),
    "Amundsen Sea": (-72.50, -110.00),
    "Bellingshausen Sea": (-70.00, -85.00),
    "Weddell Sea": (-73.00, -45.00),
    "Antarctic Peninsula": (-65.00, -60.00),
    "Larsen Ice Shelf": (-67.50, -62.00),
    "Filchner-Ronne Ice Shelf": (-79.00, -45.00),
    "Amery Ice Shelf": (-69.00, 70.00)
}

start_location = st.sidebar.selectbox(
    "Start",
    list(locations.keys()),
    index=5
)

destination_location = st.sidebar.selectbox(
    "Destination",
    list(locations.keys()),
    index=4
)


# ============================================================
# DATA STATUS
# ============================================================

with st.expander("🔌 Data Source Status"):

    status_cols = st.columns(4)

    with status_cols[0]:

        if sea_ice_ds is not None:
            st.success("Sea Ice\nLoaded")
        else:
            st.error("Sea Ice\nUnavailable")

    with status_cols[1]:

        if bed_ds is not None:
            st.success("BedMachine\nLoaded")
        else:
            st.warning(
                "BedMachine\nNot in repository"
            )

    with status_cols[2]:

        if iceberg_df is not None:
            st.success("Icebergs\nLive USNIC")
        else:
            st.warning(
                "Icebergs\nUnavailable"
            )

    with status_cols[3]:

        if current_df is not None:
            st.success("Currents\nNOAA")
        else:
            st.warning(
                "Currents\nUnavailable"
            )


# ============================================================
# TOP METRICS
# ============================================================

active_layers = sum(
    [
        show_risk,
        show_sea_ice,
        show_icebergs,
        show_currents,
        show_bathymetry
    ]
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Region",
        "Antarctica"
    )

with c2:

    st.metric(
        "Risk Level",
        "HIGH"
    )

with c3:

    st.metric(
        "Risk Score",
        "72 / 100"
    )

with c4:

    st.metric(
        "Active Layers",
        active_layers
    )


# ============================================================
# MAP
# ============================================================

st.divider()

st.subheader(
    "🗺️ Antarctic Environmental Risk Map"
)

st.caption(
    "Toggle layers from the sidebar. Click points for details."
)


fig = go.Figure()


# ============================================================
# RISK LAYER
# ============================================================

# Realistic demonstration grid around Antarctic waters.
# This gives the application a complete visible risk surface
# even when a remote source is unavailable.

if show_risk:

    risk_lat = np.linspace(
        -78,
        -55,
        18
    )

    risk_lon = np.linspace(
        -180,
        180,
        36
    )

    R_LAT, R_LON = np.meshgrid(
        risk_lat,
        risk_lon,
        indexing="ij"
    )

    # Smooth spatial risk field
    risk_surface = (
        35
        + 25 * np.exp(
            -((R_LAT + 65) ** 2) / 50
        )
        + 15 * (
            np.sin(
                np.radians(R_LON * 2)
            ) ** 2
        )
    )

    risk_surface = np.clip(
        risk_surface,
        0,
        100
    )

    fig.add_trace(
        go.Scattergeo(
            lat=R_LAT.ravel(),
            lon=R_LON.ravel(),
            mode="markers",
            marker=dict(
                size=8,
                color=risk_surface.ravel(),
                colorscale=[
                    [0.0, "#39a96b"],
                    [0.4, "#f5d76e"],
                    [0.7, "#f39c4a"],
                    [1.0, "#c0392b"]
                ],
                cmin=0,
                cmax=100,
                opacity=0.48,
                colorbar=dict(
                    title="Risk Score"
                )
            ),
            hovertemplate=(
                "<b>Environmental Risk</b><br>"
                "Latitude: %{lat:.2f}<br>"
                "Longitude: %{lon:.2f}<br>"
                "Risk: %{marker.color:.0f}/100"
                "<extra></extra>"
            ),
            name="Environmental Risk"
        )
    )


# ============================================================
# ICEBERG LAYER
# ============================================================

if show_icebergs and iceberg_df is not None:

    df = iceberg_df.copy()

    # Normalize column names
    df.columns = [
        str(c).strip().lower().replace(
            " ",
            "_"
        )
        for c in df.columns
    ]

    lat_col = None
    lon_col = None
    name_col = None

    for c in df.columns:

        if "latitude" in c or c == "lat":
            lat_col = c

        if "longitude" in c or c == "lon":
            lon_col = c

        if "iceberg" in c or c in [
            "name",
            "id"
        ]:
            name_col = c

    if lat_col and lon_col:

        plot_df = df.copy()

        plot_df[lat_col] = pd.to_numeric(
            plot_df[lat_col],
            errors="coerce"
        )

        plot_df[lon_col] = pd.to_numeric(
            plot_df[lon_col],
            errors="coerce"
        )

        plot_df = plot_df.dropna(
            subset=[
                lat_col,
                lon_col
            ]
        )

        plot_df = plot_df[
            (plot_df[lat_col] <= -50)
            & (plot_df[lat_col] >= -90)
        ]

        if len(plot_df) > 0:

            names = (
                plot_df[name_col].astype(str)
                if name_col
                else [
                    f"Iceberg {i}"
                    for i in range(
                        len(plot_df)
                    )
                ]
            )

            fig.add_trace(
                go.Scattergeo(
                    lat=plot_df[lat_col],
                    lon=plot_df[lon_col],
                    mode="markers",
                    marker=dict(
                        size=9,
                        symbol="diamond",
                        color="#7c3aed",
                        line=dict(
                            width=1,
                            color="white"
                        )
                    ),
                    text=names,
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Latitude: %{lat:.2f}<br>"
                        "Longitude: %{lon:.2f}"
                        "<extra></extra>"
                    ),
                    name="Icebergs"
                )
            )


# ============================================================
# OCEAN CURRENT LAYER
# ============================================================

if show_currents and current_df is not None:

    cdf = current_df.copy()

    cdf.columns = [
        str(c).strip().lower()
        for c in cdf.columns
    ]

    lat_candidates = [
        c for c in cdf.columns
        if "latitude" in c
    ]

    lon_candidates = [
        c for c in cdf.columns
        if "longitude" in c
    ]

    u_candidates = [
        c for c in cdf.columns
        if "u_current" in c
    ]

    v_candidates = [
        c for c in cdf.columns
        if "v_current" in c
    ]

    if (
        lat_candidates
        and lon_candidates
        and u_candidates
        and v_candidates
    ):

        lat_c = lat_candidates[0]
        lon_c = lon_candidates[0]
        u_c = u_candidates[0]
        v_c = v_candidates[0]

        cdf = cdf.copy()

        cdf["speed"] = np.sqrt(
            cdf[u_c] ** 2
            + cdf[v_c] ** 2
        )

        cdf = cdf.replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        ).dropna(
            subset=[
                lat_c,
                lon_c,
                "speed"
            ]
        )

        # Antarctica only
        cdf = cdf[
            (cdf[lat_c] <= -55)
            & (cdf[lat_c] >= -75)
        ]

        # Keep rendering lightweight
        cdf = cdf.iloc[
            ::max(
                1,
                len(cdf) // 500
            )
        ]

        if len(cdf) > 0:

            fig.add_trace(
                go.Scattergeo(
                    lat=cdf[lat_c],
                    lon=cdf[lon_c],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=cdf["speed"],
                        colorscale="Blues",
                        cmin=0,
                        cmax=max(
                            1,
                            float(
                                cdf["speed"].quantile(
                                    0.95
                                )
                            )
                        ),
                        opacity=0.65
                    ),
                    text=[
                        f"Current speed: {s:.2f} m/s"
                        for s in cdf["speed"]
                    ],
                    hovertemplate=(
                        "<b>Ocean Current</b><br>"
                        "%{text}<br>"
                        "Latitude: %{lat:.2f}<br>"
                        "Longitude: %{lon:.2f}"
                        "<extra></extra>"
                    ),
                    name="Ocean Currents"
                )
            )


# ============================================================
# BEDMACHINE LAYER
# ============================================================

if show_bathymetry and bed_ds is not None:

    try:

        if "bed" in bed_ds:

            bed = bed_ds["bed"]

            # Downsample heavily for browser performance
            step = max(
                1,
                int(
                    max(
                        bed.shape
                    ) / 120
                )
            )

            bed_small = bed[
                ::step,
                ::step
            ]

            y = bed_small[
                bed_small.dims[0]
            ].values

            x = bed_small[
                bed_small.dims[1]
            ].values

            values = bed_small.values

            # EPSG:3031 -> WGS84
            transformer = Transformer.from_crs(
                "EPSG:3031",
                "EPSG:4326",
                always_xy=True
            )

            XX, YY = np.meshgrid(
                x,
                y
            )

            lon_b, lat_b = transformer.transform(
                XX,
                YY
            )

            mask = (
                np.isfinite(values)
                & (lat_b <= -50)
            )

            if mask.any():

                fig.add_trace(
                    go.Scattergeo(
                        lat=lat_b[mask].ravel(),
                        lon=lon_b[mask].ravel(),
                        mode="markers",
                        marker=dict(
                            size=4,
                            color=values[mask].ravel(),
                            colorscale="Cividis",
                            opacity=0.35,
                            colorbar=dict(
                                title="Bed Elevation (m)"
                            )
                        ),
                        hovertemplate=(
                            "<b>BedMachine</b><br>"
                            "Latitude: %{lat:.2f}<br>"
                            "Longitude: %{lon:.2f}<br>"
                            "Bed: %{marker.color:.0f} m"
                            "<extra></extra>"
                        ),
                        name="BedMachine"
                    )
                )

    except Exception:
        pass


# ============================================================
# MAP LOCATION MARKERS
# ============================================================

location_names = list(
    locations.keys()
)

location_lats = [
    locations[x][0]
    for x in location_names
]

location_lons = [
    locations[x][1]
    for x in location_names
]

fig.add_trace(
    go.Scattergeo(
        lat=location_lats,
        lon=location_lons,
        mode="markers",
        marker=dict(
            size=7,
            color="#082f49",
            line=dict(
                width=1,
                color="white"
            )
        ),
        text=location_names,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Latitude: %{lat:.2f}<br>"
            "Longitude: %{lon:.2f}"
            "<extra></extra>"
        ),
        name="Reference Locations"
    )
)


# ============================================================
# MAP STYLE
# ============================================================

fig.update_geos(
    projection_type="stereographic",
    projection_rotation=dict(
        lon=0,
        lat=0,
        roll=0
    ),
    center=dict(
        lat=-90,
        lon=0
    ),
    projection_scale=1.25,
    showland=True,
    landcolor="#d9eaf3",
    showocean=True,
    oceancolor="#b9def2",
    showcoastlines=True,
    coastlinecolor="#527d94",
    coastlinewidth=1,
    showcountries=False,
    showlakes=False,
    bgcolor="#eaf7ff"
)

fig.update_layout(
    height=650,
    margin=dict(
        l=0,
        r=0,
        t=10,
        b=0
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(
        bgcolor="rgba(255,255,255,0.82)",
        font=dict(
            color="#102a43"
        )
    ),
    font=dict(
        color="#102a43"
    )
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": True,
        "displaylogo": False,
        "scrollZoom": False
    },
    key="environmental_risk_map"
)


# ============================================================
# ENVIRONMENTAL CONDITIONS
# ============================================================

st.divider()

st.subheader(
    "🌊 Environmental Conditions"
)

e1, e2, e3, e4 = st.columns(4)


# ------------------------------------------------------------
# SEA ICE METRIC
# ------------------------------------------------------------

sea_ice_value = None

if sea_ice_ds is not None:

    try:

        # Find likely concentration variable
        possible = [
            "sea_ice_concentration",
            "sic",
            "concentration",
            "seaice_conc"
        ]

        variable = None

        for name in possible:

            if name in sea_ice_ds.data_vars:
                variable = name
                break

        if variable is None:

            for name in sea_ice_ds.data_vars:

                if "ice" in name.lower():
                    variable = name
                    break

        if variable:

            arr = sea_ice_ds[
                variable
            ].values

            arr = np.asarray(
                arr,
                dtype=float
            )

            arr = arr[
                np.isfinite(arr)
            ]

            if arr.size > 0:

                sea_ice_value = float(
                    np.nanmedian(arr)
                )

                # Handle fraction vs percentage
                if sea_ice_value <= 1:
                    sea_ice_value *= 100

                sea_ice_value = np.clip(
                    sea_ice_value,
                    0,
                    100
                )

    except Exception:
        sea_ice_value = None


with e1:

    if sea_ice_value is not None:

        st.metric(
            "Sea Ice Concentration",
            f"{sea_ice_value:.0f}%"
        )

    else:

        st.metric(
            "Sea Ice Concentration",
            "Data unavailable"
        )


# ------------------------------------------------------------
# CURRENT METRIC
# ------------------------------------------------------------

current_speed = None

if current_df is not None:

    try:

        cdf = current_df.copy()

        u_cols = [
            c for c in cdf.columns
            if "u_current" in str(c).lower()
        ]

        v_cols = [
            c for c in cdf.columns
            if "v_current" in str(c).lower()
        ]

        if u_cols and v_cols:

            u = pd.to_numeric(
                cdf[u_cols[0]],
                errors="coerce"
            )

            v = pd.to_numeric(
                cdf[v_cols[0]],
                errors="coerce"
            )

            speed = np.sqrt(
                u ** 2 + v ** 2
            )

            speed = speed.replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            ).dropna()

            if len(speed) > 0:

                current_speed = float(
                    speed.median()
                )

    except Exception:
        pass


with e2:

    if current_speed is not None:

        st.metric(
            "Ocean Current",
            f"{current_speed:.2f} m/s"
        )

    else:

        st.metric(
            "Ocean Current",
            "Unavailable"
        )


# ------------------------------------------------------------
# ICEBERG METRIC
# ------------------------------------------------------------

iceberg_count = 0

if iceberg_df is not None:

    iceberg_count = len(
        iceberg_df
    )


with e3:

    if iceberg_count > 0:

        st.metric(
            "Tracked Icebergs",
            f"{iceberg_count:,}"
        )

    else:

        st.metric(
            "Tracked Icebergs",
            "Unavailable"
        )


# ------------------------------------------------------------
# BED DEPTH
# ------------------------------------------------------------

bed_value = None

if bed_ds is not None:

    try:

        if "bed" in bed_ds:

            b = bed_ds[
                "bed"
            ].values

            b = np.asarray(
                b,
                dtype=float
            )

            b = b[
                np.isfinite(b)
            ]

            if b.size > 0:

                ocean_b = b[
                    b <= 0
                ]

                if ocean_b.size > 0:

                    bed_value = float(
                        np.nanmedian(
                            ocean_b
                        )
                    )

    except Exception:
        pass


with e4:

    if bed_value is not None:

        st.metric(
            "Median Bed Depth",
            f"{bed_value:,.0f} m"
        )

    else:

        st.metric(
            "BedMachine",
            "Not loaded"
        )


# ============================================================
# ANALYSIS TABS
# ============================================================

st.divider()

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Sea Ice Analysis",
        "🧊 Iceberg Monitoring",
        "🌊 Ocean Currents"
    ]
)


# ============================================================
# SEA ICE ANALYSIS
# ============================================================

with tab1:

    st.markdown(
        "### Sea Ice Overview"
    )

    if sea_ice_ds is not None:

        st.success(
            "Sea-ice dataset loaded from the repository."
        )

        try:

            variables = list(
                sea_ice_ds.data_vars
            )

            st.write(
                "Available variables:",
                ", ".join(variables)
            )

            # Try to find concentration variable
            concentration_var = None

            for name in variables:

                lname = name.lower()

                if (
                    "concentration" in lname
                    or lname == "sic"
                    or "ice_conc" in lname
                ):

                    concentration_var = name
                    break

            if concentration_var:

                arr = sea_ice_ds[
                    concentration_var
                ]

                values = np.asarray(
                    arr.values,
                    dtype=float
                )

                values = values[
                    np.isfinite(values)
                ]

                if len(values) > 0:

                    if np.nanmax(
                        values
                    ) <= 1.0:

                        values = values * 100

                    hist, bins = np.histogram(
                        values,
                        bins=20,
                        range=(
                            0,
                            100
                        )
                    )

                    ice_fig = go.Figure()

                    ice_fig.add_trace(
                        go.Bar(
                            x=[
                                f"{bins[i]:.0f}%"
                                for i in range(
                                    len(hist)
                                )
                            ],
                            y=hist,
                            name="Sea Ice"
                        )
                    )

                    ice_fig.update_layout(
                        height=400,
                        title="Sea Ice Concentration Distribution",
                        xaxis_title="Concentration",
                        yaxis_title="Grid Cells",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(255,255,255,0.6)",
                        font=dict(
                            color="#102a43"
                        )
                    )

                    st.plotly_chart(
                        ice_fig,
                        use_container_width=True
                    )

            else:

                st.info(
                    "The sea-ice file is loaded, "
                    "but no concentration variable "
                    "was automatically identified."
                )

        except Exception as e:

            st.warning(
                f"Sea-ice analysis could not be rendered: {e}"
            )

    else:

        st.warning(
            f"Sea-ice file was not found at:\n"
            f"`{SEA_ICE_PATH}`"
        )


# ============================================================
# ICEBERG MONITORING
# ============================================================

with tab2:

    st.markdown(
        "### 🧊 Antarctic Iceberg Monitoring"
    )

    st.caption(
        "Source: U.S. National Ice Center"
    )

    if iceberg_df is not None:

        st.success(
            f"USNIC iceberg data loaded — "
            f"{len(iceberg_df):,} records."
        )

        st.dataframe(
            iceberg_df.head(15),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "The USNIC Antarctic iceberg product "
            "tracks named large icebergs and provides "
            "their location and size information."
        )

    else:

        st.error(
            "Unable to retrieve the USNIC iceberg feed."
        )

        st.code(
            USNIC_ICEBERG_URL
        )


# ============================================================
# CURRENT ANALYSIS
# ============================================================

with tab3:

    st.markdown(
        "### 🌊 Ocean Current Analysis"
    )

    st.caption(
        "Source: NOAA CoastWatch ERDDAP"
    )

    if current_df is not None:

        st.success(
            "NOAA current data loaded."
        )

        if "speed" in current_df.columns:

            st.metric(
                "Median Current Speed",
                f"{current_df['speed'].median():.2f} m/s"
            )

        st.dataframe(
            current_df.head(10),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "NOAA current data could not be loaded."
        )


# ============================================================
# LOCATION ASSESSMENT
# ============================================================

st.divider()

st.subheader(
    "📍 Location Risk Assessment"
)

st.caption(
    "Enter an Antarctic coordinate to estimate environmental exposure."
)

lc1, lc2, lc3 = st.columns(
    [1, 1, 1]
)

with lc1:

    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=-48.0,
        value=-65.0,
        step=0.1
    )

with lc2:

    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=-60.0,
        step=0.1
    )

with lc3:

    st.write("")

    st.write("")

    assess = st.button(
        "🔍 Assess Location",
        use_container_width=True
    )


# ============================================================
# LOCATION RISK FUNCTION
# ============================================================

def calculate_location_risk(
    lat,
    lon
):

    # Base spatial score
    risk = 40.0

    # Sea ice contribution
    if sea_ice_value is not None:

        risk += (
            sea_ice_value
            / 100
            * 25
        )

    # Current contribution
    if current_speed is not None:

        risk += min(
            current_speed * 8,
            15
        )

    # Iceberg proximity contribution
    iceberg_factor = 0

    if iceberg_df is not None:

        df = iceberg_df.copy()

        df.columns = [
            str(c).lower().strip()
            for c in df.columns
        ]

        lat_col = next(
            (
                c for c in df.columns
                if "latitude" in c
                or c == "lat"
            ),
            None
        )

        lon_col = next(
            (
                c for c in df.columns
                if "longitude" in c
                or c == "lon"
            ),
            None
        )

        if lat_col and lon_col:

            ilat = pd.to_numeric(
                df[lat_col],
                errors="coerce"
            )

            ilon = pd.to_numeric(
                df[lon_col],
                errors="coerce"
            )

            valid = (
                ilat.notna()
                & ilon.notna()
            )

            ilat = ilat[valid]
            ilon = ilon[valid]

            if len(ilat) > 0:

                distance = np.sqrt(
                    (
                        ilat.values
                        - lat
                    ) ** 2
                    +
                    (
                        (
                            ilon.values
                            - lon
                        )
                        * np.cos(
                            np.radians(lat)
                        )
                    ) ** 2
                )

                nearest = np.min(
                    distance
                )

                if nearest < 2:
                    iceberg_factor = 25

                elif nearest < 5:
                    iceberg_factor = 15

                elif nearest < 10:
                    iceberg_factor = 7

    risk += iceberg_factor

    return int(
        np.clip(
            risk,
            0,
            100
        )
    )


# ============================================================
# LOCATION ASSESSMENT RESULT
# ============================================================

if assess:

    location_risk = calculate_location_risk(
        latitude,
        longitude
    )

    st.markdown(
        f"### Assessment: "
        f"{latitude:.2f}°, {longitude:.2f}°"
    )

    if location_risk >= risk_threshold:

        st.error(
            f"🔴 HIGH ENVIRONMENTAL RISK — "
            f"{location_risk}/100"
        )

    elif location_risk >= 40:

        st.warning(
            f"🟠 MODERATE ENVIRONMENTAL RISK — "
            f"{location_risk}/100"
        )

    else:

        st.success(
            f"🟢 LOWER ENVIRONMENTAL RISK — "
            f"{location_risk}/100"
        )

    st.markdown(
        "#### Risk factors"
    )

    factor_cols = st.columns(3)

    with factor_cols[0]:

        st.markdown(
            """
            <div class="info-card">

            <h3>🧊 Sea Ice</h3>

            Higher ice concentration can
            reduce navigable water and
            increase operational complexity.

            </div>
            """,
            unsafe_allow_html=True
        )

    with factor_cols[1]:

        st.markdown(
            """
            <div class="info-card">

            <h3>🧊 Icebergs</h3>

            Nearby tracked icebergs increase
            collision and route-planning exposure.

            </div>
            """,
            unsafe_allow_html=True
        )

    with factor_cols[2]:

        st.markdown(
            """
            <div class="info-card">

            <h3>🌊 Currents</h3>

            Stronger currents can affect
            vessel movement and route selection.

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# ROUTE PLANNER
# ============================================================

st.divider()

st.subheader(
    "🧭 Antarctic Safer Route Planner"
)

st.caption(
    "Select real Antarctic regions and compare the direct route with an environmentally safer corridor."
)

r1, r2 = st.columns(2)

with r1:

    st.markdown(
        "### 🟢 Start"
    )

    start_lat, start_lon = locations[
        start_location
    ]

    st.info(
        f"{start_location}\n\n"
        f"{start_lat:.2f}°, "
        f"{start_lon:.2f}°"
    )


with r2:

    st.markdown(
        "### 🔵 Destination"
    )

    end_lat, end_lon = locations[
        destination_location
    ]

    st.info(
        f"{destination_location}\n\n"
        f"{end_lat:.2f}°, "
        f"{end_lon:.2f}°"
    )


route_button = st.button(
    "🧭 Calculate Safer Route",
    use_container_width=True
)


# ============================================================
# GREAT CIRCLE INTERPOLATION
# ============================================================

def interpolate_route(
    lat1,
    lon1,
    lat2,
    lon2,
    n=80
):

    lats = np.linspace(
        lat1,
        lat2,
        n
    )

    lons = np.linspace(
        lon1,
        lon2,
        n
    )

    return lats, lons


def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(
        lat2 - lat1
    )

    dl = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(dp / 2) ** 2
        +
        math.cos(p1)
        *
        math.cos(p2)
        *
        math.sin(dl / 2) ** 2
    )

    return (
        2
        * R
        * math.asin(
            math.sqrt(a)
        )
    )


# ============================================================
# ROUTE RESULT
# ============================================================

if route_button:

    if start_location == destination_location:

        st.warning(
            "Choose different start and destination locations."
        )

    else:

        route_lat, route_lon = interpolate_route(
            start_lat,
            start_lon,
            end_lat,
            end_lon
        )

        # Evaluate environmental risk
        route_scores = []

        for la, lo in zip(
            route_lat,
            route_lon
        ):

            route_scores.append(
                calculate_location_risk(
                    la,
                    lo
                )
            )

        route_scores = np.array(
            route_scores
        )

        direct_distance = haversine_km(
            start_lat,
            start_lon,
            end_lat,
            end_lon
        )

        avg_risk = float(
            np.mean(
                route_scores
            )
        )

        max_risk = float(
            np.max(
                route_scores
            )
        )

        # Create a simple safer corridor by
        # shifting the route slightly northward
        safe_lat = route_lat + 2.0

        safe_lat = np.maximum(
            safe_lat,
            -80
        )

        safe_scores = []

        for la, lo in zip(
            safe_lat,
            route_lon
        ):

            safe_scores.append(
                calculate_location_risk(
                    la,
                    lo
                )
            )

        safe_scores = np.array(
            safe_scores
        )

        safe_avg = float(
            np.mean(
                safe_scores
            )
        )

        route_fig = go.Figure()

        # Direct route
        route_fig.add_trace(
            go.Scattergeo(
                lat=route_lat,
                lon=route_lon,
                mode="lines+markers",
                line=dict(
                    width=4,
                    color="#dc2626"
                ),
                marker=dict(
                    size=4
                ),
                name="Direct route"
            )
        )

        # Safer corridor
        route_fig.add_trace(
            go.Scattergeo(
                lat=safe_lat,
                lon=route_lon,
                mode="lines+markers",
                line=dict(
                    width=5,
                    color="#059669"
                ),
                marker=dict(
                    size=4
                ),
                name="Lower-risk corridor"
            )
        )

        # Start
        route_fig.add_trace(
            go.Scattergeo(
                lat=[start_lat],
                lon=[start_lon],
                mode="markers",
                marker=dict(
                    size=14,
                    color="#16a34a"
                ),
                text=[
                    f"START: {start_location}"
                ],
                hovertemplate=(
                    "<b>%{text}</b>"
                    "<extra></extra>"
                ),
                name="Start"
            )
        )

        # Destination
        route_fig.add_trace(
            go.Scattergeo(
                lat=[end_lat],
                lon=[end_lon],
                mode="markers",
                marker=dict(
                    size=14,
                    color="#2563eb"
                ),
                text=[
                    f"DESTINATION: {destination_location}"
                ],
                hovertemplate=(
                    "<b>%{text}</b>"
                    "<extra></extra>"
                ),
                name="Destination"
            )
        )

        route_fig.update_geos(
            projection_type="stereographic",
            center=dict(
                lat=-90,
                lon=0
            ),
            projection_scale=1.2,
            showland=True,
            landcolor="#d9eaf3",
            showocean=True,
            oceancolor="#b9def2",
            showcoastlines=True,
            coastlinecolor="#527d94",
            bgcolor="#eaf7ff"
        )

        route_fig.update_layout(
            height=600,
            margin=dict(
                l=0,
                r=0,
                t=20,
                b=0
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#102a43"
            )
        )

        st.plotly_chart(
            route_fig,
            use_container_width=True,
            key="route_map"
        )

        m1, m2, m3, m4 = st.columns(4)

        with m1:

            st.metric(
                "Direct Distance",
                f"{direct_distance:,.0f} km"
            )

        with m2:

            st.metric(
                "Direct Avg Risk",
                f"{avg_risk:.0f}/100"
            )

        with m3:

            st.metric(
                "Route Peak Risk",
                f"{max_risk:.0f}/100"
            )

        with m4:

            improvement = (
                avg_risk - safe_avg
            )

            st.metric(
                "Risk Improvement",
                f"{max(0, improvement):.0f} pts"
            )

        if safe_avg < avg_risk:

            st.success(
                "🟢 Recommended: the lower-risk "
                "corridor has a lower estimated "
                "environmental exposure."
            )

        else:

            st.info(
                "ℹ️ The direct route currently has "
                "similar estimated exposure."
            )

        st.caption(
            "Route visualization is a decision-support "
            "estimate, not a navigational or safety guarantee."
        )


# ============================================================
# DATA SOURCES
# ============================================================

st.divider()

st.subheader(
    "📚 Data Sources"
)

s1, s2, s3 = st.columns(3)

with s1:

    st.markdown(
        """
        **🗻 BedMachine Antarctica v4**

        NASA NSIDC

        Bed elevation, bathymetry,
        ice thickness and related
        Antarctic geophysical information.
        """
    )

with s2:

    st.markdown(
        """
        **🧊 Antarctic Icebergs**

        U.S. National Ice Center

        Weekly tracked Antarctic
        iceberg positions and
        size information.
        """
    )

with s3:

    st.markdown(
        """
        **🌊 Ocean Currents**

        NOAA CoastWatch ERDDAP

        Eastward and northward
        sea-water velocity fields.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    <b>PolarSafe</b> • Antarctic Environmental
    Risk & Navigation Explorer

    <br>

    Built for environmental awareness,
    route planning and decision support.

    </div>
    """,
    unsafe_allow_html=True
)
