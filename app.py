import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# POLAR-SAFE
# Antarctic Environmental Risk & Navigation Dashboard
# SIH26059
# ============================================================

st.set_page_config(
    page_title="POLAR-SAFE | Antarctic Risk Explorer",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
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
            radial-gradient(
                circle at 15% 0%,
                rgba(41, 117, 181, 0.20),
                transparent 35%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(20, 93, 145, 0.18),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #eef7fc 0%,
                #dceefa 45%,
                #f7fbfe 100%
            );

        color: #102a43;
    }

    .main {
        background: transparent;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }

    /* ---------- SIDEBAR ---------- */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #08243d 0%,
                #0b3557 48%,
                #0e426b 100%
            );
    }

    [data-testid="stSidebar"] * {
        color: #f4faff !important;
    }

    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        color: #e8f5ff !important;
        font-weight: 600;
    }

    /* ---------- HEADINGS ---------- */

    h1, h2, h3 {
        color: #0a2c47 !important;
    }

    p, li, label {
        color: #183b56;
    }

    /* ---------- HERO ---------- */

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 22px;

        background:
            linear-gradient(
                135deg,
                #082b49 0%,
                #0c466f 50%,
                #12658e 100%
            );

        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 0 14px 40px rgba(8,43,73,0.18);

        margin-bottom: 1.2rem;
    }

    .hero-title {
        color: #ffffff !important;
        font-size: 2.7rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        color: #d9f1ff !important;
        font-size: 1.05rem;
        margin-top: 0.15rem;
    }

    .hero-tag {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        color: #ffffff !important;
        font-size: 0.85rem;
        border: 1px solid rgba(255,255,255,0.15);
    }

    /* ---------- SECTION ---------- */

    .section-title {
        color: #083452 !important;
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 1rem;
        margin-bottom: 0.4rem;
    }

    .section-subtitle {
        color: #46647a !important;
        margin-bottom: 1rem;
    }

    /* ---------- CARDS ---------- */

    .info-card {
        background: rgba(255,255,255,0.88);
        border: 1px solid #c9e1ef;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 6px 22px rgba(14,66,107,0.08);
        min-height: 125px;
    }

    .info-card h3 {
        margin-top: 0;
        color: #0a3b5c !important;
        font-size: 1.05rem;
    }

    .info-card p {
        color: #49677a !important;
        font-size: 0.92rem;
    }

    /* ---------- RISK CARDS ---------- */

    .risk-high {
        background: linear-gradient(135deg, #fff1f1, #ffe0e0);
        border: 1px solid #f2aaaa;
        color: #7f1d1d !important;
        padding: 1.2rem;
        border-radius: 16px;
        text-align: center;
    }

    .risk-medium {
        background: linear-gradient(135deg, #fff9e8, #ffefbf);
        border: 1px solid #e7cc75;
        color: #713f12 !important;
        padding: 1.2rem;
        border-radius: 16px;
        text-align: center;
    }

    .risk-low {
        background: linear-gradient(135deg, #ecfdf5, #d3f6e5);
        border: 1px solid #9ad8b7;
        color: #14532d !important;
        padding: 1.2rem;
        border-radius: 16px;
        text-align: center;
    }

    /* ---------- WARNING ---------- */

    .warning-box {
        padding: 0.95rem 1.2rem;
        border-radius: 14px;
        background: #e5f4fc;
        border: 1px solid #a9d5eb;
        color: #12415e !important;
        margin-bottom: 1rem;
    }

    /* ---------- ROUTE ---------- */

    .route-card {
        padding: 1.3rem;
        border-radius: 16px;
        border: 1px solid #b9d8e9;
        background:
            linear-gradient(
                135deg,
                #ffffff,
                #eaf6fc
            );
        margin-bottom: 0.8rem;
        box-shadow: 0 6px 20px rgba(8,43,73,0.07);
    }

    .route-card h3 {
        color: #0b4265 !important;
        margin-top: 0;
    }

    .route-card p {
        color: #426174 !important;
    }

    /* ---------- BUTTONS ---------- */

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #8bbdd6;
        background: linear-gradient(
            135deg,
            #0b4f78,
            #0d6b96
        );
        color: white !important;
        font-weight: 700;
        min-height: 2.6rem;
    }

    .stButton > button:hover {
        background: linear-gradient(
            135deg,
            #083d60,
            #095a82
        );
        color: white !important;
        border-color: #6ca9c7;
    }

    /* ---------- DATAFRAME ---------- */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---------- METRICS ---------- */

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.88);
        border: 1px solid #c7dfed;
        padding: 0.9rem;
        border-radius: 14px;
        box-shadow: 0 5px 18px rgba(8,43,73,0.06);
    }

    [data-testid="stMetricLabel"] {
        color: #426276 !important;
    }

    [data-testid="stMetricValue"] {
        color: #073a5a !important;
    }

    /* ---------- TABS ---------- */

    button[data-baseweb="tab"] {
        color: #244c65 !important;
        font-weight: 700;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #075985 !important;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #557286 !important;
        padding: 1.2rem;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "sea_ice": os.path.join(
        BASE_DIR,
        "antarctic_sea_ice_risk_2025-04-16.csv"
    ),

    "iceberg_risk": os.path.join(
        BASE_DIR,
        "iceberg_risk_layer.csv"
    ),

    "iceberg_trajectory": os.path.join(
        BASE_DIR,
        "iceberg_trajectory_predictions.csv"
    ),

    "combined": os.path.join(
        BASE_DIR,
        "antarctic_combined_navigation_risk_2025-04-16.csv"
    ),

    "shortest": os.path.join(
        BASE_DIR,
        "shortest_route.csv"
    ),

    "balanced": os.path.join(
        BASE_DIR,
        "balanced_route.csv"
    ),

    "risk_aware": os.path.join(
        BASE_DIR,
        "risk_aware_route.csv"
    ),

    "comparison": os.path.join(
        BASE_DIR,
        "route_comparison.csv"
    ),

    "dashboard": os.path.join(
        BASE_DIR,
        "dashboard_data.json"
    ),
}


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_csv(path):

    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(
            f"Could not read {os.path.basename(path)}: {exc}"
        )
        return pd.DataFrame()


@st.cache_data
def load_json(path):

    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return {}


sea_ice = load_csv(PATHS["sea_ice"])
iceberg_risk = load_csv(PATHS["iceberg_risk"])
iceberg_trajectory = load_csv(PATHS["iceberg_trajectory"])
combined = load_csv(PATHS["combined"])

shortest_route = load_csv(PATHS["shortest"])
balanced_route = load_csv(PATHS["balanced"])
risk_aware_route = load_csv(PATHS["risk_aware"])
route_comparison = load_csv(PATHS["comparison"])

dashboard_data = load_json(PATHS["dashboard"])


# ============================================================
# CLEAN COORDINATES
# ============================================================

def clean_coordinates(df):

    if df.empty:
        return df

    out = df.copy()

    for col in ("latitude", "longitude"):

        if col in out.columns:
            out[col] = pd.to_numeric(
                out[col],
                errors="coerce"
            )

    if "longitude" in out.columns:

        out["longitude"] = (
            (out["longitude"] + 180) % 360
        ) - 180

    keep = [
        c for c in
        ("latitude", "longitude")
        if c in out.columns
    ]

    if keep:
        out = out.dropna(subset=keep)

    return out


sea_ice = clean_coordinates(sea_ice)
iceberg_risk = clean_coordinates(iceberg_risk)
iceberg_trajectory = clean_coordinates(iceberg_trajectory)
combined = clean_coordinates(combined)

shortest_route = clean_coordinates(shortest_route)
balanced_route = clean_coordinates(balanced_route)
risk_aware_route = clean_coordinates(risk_aware_route)


# ============================================================
# HELPERS
# ============================================================

def first_existing_column(df, names):

    for name in names:

        if name in df.columns:
            return name

    return None


def risk_column(df):

    return first_existing_column(
        df,
        [
            "combined_risk_score",
            "sea_ice_risk_score",
            "iceberg_risk_score",
            "risk_score",
        ],
    )


def route_distance(df):

    col = first_existing_column(
        df,
        [
            "distance_km",
            "route_distance_km",
            "cumulative_distance_km",
            "total_distance_km",
        ],
    )

    if col is None or df.empty:
        return np.nan

    values = pd.to_numeric(
        df[col],
        errors="coerce"
    ).dropna()

    return (
        float(values.max())
        if not values.empty
        else np.nan
    )


def sample_map_data(df, max_points=25000):

    """
    Large CSV layers can contain hundreds of thousands
    of points. Rendering all of them causes browser
    lag/blinking.

    We therefore display a representative subset.
    The original CSV remains untouched.
    """

    if len(df) <= max_points:
        return df

    return df.sample(
        n=max_points,
        random_state=42
    )


# ============================================================
# POLAR MAP SETTINGS
# ============================================================

def polar_geo_settings():

    return dict(

        projection=dict(
            type="stereographic"
        ),

        center=dict(
            lat=-90,
            lon=0
        ),

        projection_rotation=dict(
            lon=0,
            lat=-90,
            roll=0
        ),

        showland=True,
        landcolor="#d9eaf3",

        showocean=True,
        oceancolor="#d7edf8",

        showcountries=True,
        countrycolor="#8aaabd",

        showcoastlines=True,
        coastlinecolor="#31566b",
        coastlinewidth=0.8,

        showframe=False,

        lataxis=dict(
            showgrid=True,
            gridcolor="#b5cfdd",
            gridwidth=0.5
        ),

        lonaxis=dict(
            showgrid=True,
            gridcolor="#b5cfdd",
            gridwidth=0.5
        ),

        bgcolor="#e8f5fb",
    )


# ============================================================
# MAP CREATOR
# ============================================================

def create_risk_map(
    df,
    risk_col,
    title,
    marker_size=6,
    colorscale="Turbo",
):

    fig = go.Figure()

    if df.empty:

        fig.add_annotation(
            text="No data available for this layer.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                size=18,
                color="#173b56"
            )
        )

    else:

        plot_df = sample_map_data(
            df,
            max_points=25000
        )

        if risk_col and risk_col in plot_df.columns:

            values = pd.to_numeric(
                plot_df[risk_col],
                errors="coerce"
            ).fillna(0)

            fig.add_trace(
                go.Scattergeo(

                    lon=plot_df["longitude"],
                    lat=plot_df["latitude"],

                    mode="markers",

                    marker=dict(
                        size=marker_size,
                        color=values,
                        colorscale=colorscale,
                        cmin=0,
                        cmax=100,
                        opacity=0.72,
                        line=dict(
                            width=0
                        ),

                        colorbar=dict(
                            title=dict(
                                text="Risk<br>0–100",
                                font=dict(
                                    color="#173b56"
                                )
                            ),
                            tickfont=dict(
                                color="#173b56"
                            )
                        )
                    ),

                    customdata=np.column_stack(
                        [
                            plot_df["latitude"],
                            plot_df["longitude"],
                            values
                        ]
                    ),

                    hovertemplate=(
                        "<b>Environmental Risk</b><br>"
                        "Latitude: %{customdata[0]:.2f}°<br>"
                        "Longitude: %{customdata[1]:.2f}°<br>"
                        "Risk: %{customdata[2]:.1f}/100"
                        "<extra></extra>"
                    ),

                    name="Risk"
                )
            )

        else:

            fig.add_trace(
                go.Scattergeo(

                    lon=plot_df["longitude"],
                    lat=plot_df["latitude"],

                    mode="markers",

                    marker=dict(
                        size=marker_size,
                        opacity=0.65
                    ),

                    hovertemplate=(
                        "Latitude: %{lat:.2f}°<br>"
                        "Longitude: %{lon:.2f}°"
                        "<extra></extra>"
                    ),

                    name="Observations"
                )
            )

    fig.update_geos(
        **polar_geo_settings()
    )

    fig.update_layout(

        title=dict(
            text=title,
            font=dict(
                size=20,
                color="#0a3552"
            ),
            x=0.02
        ),

        height=620,

        margin=dict(
            l=0,
            r=0,
            t=60,
            b=0
        ),

        paper_bgcolor="#e8f5fb",
        plot_bgcolor="#e8f5fb",

        font=dict(
            color="#173b56"
        ),

        legend=dict(
            bgcolor="rgba(255,255,255,0.75)",
            font=dict(
                color="#173b56"
            )
        ),

        uirevision="polar-safe-map"
    )

    return fig


# ============================================================
# ROUTE MAP
# ============================================================

def add_route_trace(
    fig,
    df,
    name,
    width
):

    if df.empty:
        return

    if not {
        "latitude",
        "longitude"
    }.issubset(df.columns):

        return

    plot_df = sample_map_data(
        df,
        max_points=5000
    )

    fig.add_trace(
        go.Scattergeo(

            lon=plot_df["longitude"],
            lat=plot_df["latitude"],

            mode="lines",

            line=dict(
                width=width
            ),

            name=name,

            hovertemplate=(
                "Latitude: %{lat:.2f}°<br>"
                "Longitude: %{lon:.2f}°"
                f"<extra>{name}</extra>"
            )
        )
    )


def create_route_map():

    fig = go.Figure()

    add_route_trace(
        fig,
        shortest_route,
        "Shortest Route",
        3
    )

    add_route_trace(
        fig,
        balanced_route,
        "Balanced Route",
        4
    )

    add_route_trace(
        fig,
        risk_aware_route,
        "Risk-Aware Route",
        5
    )

    # Bharati Station

    fig.add_trace(
        go.Scattergeo(

            lon=[76.19525],
            lat=[-69.4068],

            mode="markers+text",

            marker=dict(
                size=13,
                symbol="star",
                line=dict(
                    width=1
                )
            ),

            text=["Bharati"],
            textposition="top center",

            name="Origin",

            hovertemplate=(
                "<b>Bharati Station</b><br>"
                "Latitude: -69.4068°<br>"
                "Longitude: 76.1953°"
                "<extra></extra>"
            )
        )
    )

    # Maitri Station

    fig.add_trace(
        go.Scattergeo(

            lon=[11.73333],
            lat=[-70.76444],

            mode="markers+text",

            marker=dict(
                size=13,
                symbol="diamond",
                line=dict(
                    width=1
                )
            ),

            text=["Maitri"],
            textposition="top center",

            name="Destination",

            hovertemplate=(
                "<b>Maitri Station</b><br>"
                "Latitude: -70.7644°<br>"
                "Longitude: 11.7333°"
                "<extra></extra>"
            )
        )
    )

    fig.update_geos(
        **polar_geo_settings()
    )

    fig.update_layout(

        title=dict(
            text="Bharati → Maitri Route Comparison",
            font=dict(
                size=20,
                color="#0a3552"
            )
        ),

        height=650,

        margin=dict(
            l=0,
            r=0,
            t=60,
            b=0
        ),

        paper_bgcolor="#e8f5fb",

        font=dict(
            color="#173b56"
        ),

        legend=dict(
            bgcolor="rgba(255,255,255,0.78)",
            font=dict(
                color="#173b56"
            )
        ),

        uirevision="polar-safe-route"
    )

    return fig


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🧊 POLAR-SAFE
        </div>

        <div class="hero-subtitle">
            Antarctic Environmental Risk & Navigation Intelligence
        </div>

        <div class="hero-subtitle">
            Understand the ice • Track hazards • Choose safer routes
        </div>

        <div class="hero-tag">
            SIH26059 • Decision Support Prototype
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="warning-box">
        <b>⚠️ Decision-support prototype:</b>
        POLAR-SAFE combines environmental datasets to help identify
        potentially hazardous Antarctic marine conditions.
        It is intended to support — not replace — qualified navigation decisions.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧭 POLAR-SAFE")

st.sidebar.caption(
    "Antarctic navigation risk explorer"
)

st.sidebar.markdown("---")

layer = st.sidebar.radio(
    "Explore",
    [
        "Overview",
        "Combined Risk",
        "Sea-Ice",
        "Icebergs",
        "Routes",
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Route Scenario")

origin = st.sidebar.selectbox(
    "Origin",
    [
        "Bharati Station"
    ]
)

destination = st.sidebar.selectbox(
    "Destination",
    [
        "Maitri Station"
    ]
)

safety_preference = st.sidebar.slider(
    "Safety Preference",
    0,
    100,
    80,
    help=(
        "Higher values place greater emphasis "
        "on avoiding environmental risk."
    )
)

st.sidebar.caption(
    f"Safety Preference: {safety_preference}%"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "POLAR-SAFE • SIH26059"
)


# ============================================================
# KPI CALCULATIONS
# ============================================================

combined_col = risk_column(combined)
sea_ice_col = risk_column(sea_ice)
iceberg_col = risk_column(iceberg_risk)


if combined_col and not combined.empty:

    combined_values = pd.to_numeric(
        combined[combined_col],
        errors="coerce"
    ).dropna()

    average_risk = (
        float(combined_values.mean())
        if not combined_values.empty
        else 0
    )

    maximum_risk = (
        float(combined_values.max())
        if not combined_values.empty
        else 0
    )

else:

    average_risk = 0
    maximum_risk = 0


if "combined_risk_level" in combined.columns:

    high_risk_cells = int(
        combined["combined_risk_level"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "high",
                "very high"
            ]
        )
        .sum()
    )

elif combined_col:

    values = pd.to_numeric(
        combined[combined_col],
        errors="coerce"
    )

    high_risk_cells = int(
        (values >= 60).sum()
    )

else:

    high_risk_cells = 0


if "iceberg_id" in iceberg_risk.columns:

    tracked_icebergs = int(
        iceberg_risk["iceberg_id"]
        .nunique()
    )

else:

    tracked_icebergs = 0


# ============================================================
# KPI BAR
# ============================================================

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Average Risk",
    f"{average_risk:.1f}/100"
)

k2.metric(
    "Maximum Risk",
    f"{maximum_risk:.1f}/100"
)

k3.metric(
    "Tracked Icebergs",
    f"{tracked_icebergs:,}"
)

k4.metric(
    "High-Risk Cells",
    f"{high_risk_cells:,}"
)


st.markdown("---")


# ============================================================
# OVERVIEW
# ============================================================

if layer == "Overview":

    st.markdown(
        '<div class="section-title">🌊 Antarctic Environmental Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'A single view of the environmental factors that influence '
        'navigation risk around Antarctica.'
        '</div>',
        unsafe_allow_html=True
    )

    # Overview cards

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="info-card">

            <h3>🧊 Sea-Ice Conditions</h3>

            <p>
            Monitor sea-ice concentration and the derived
            navigation-risk surface.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="info-card">

            <h3>🧊 Iceberg Monitoring</h3>

            <p>
            Explore observed iceberg positions,
            predicted trajectories and risk levels.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="info-card">

            <h3>🧭 Route Planning</h3>

            <p>
            Compare shortest, balanced and risk-aware
            navigation routes.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("")

    # Main overview map

    if not combined.empty:

        st.plotly_chart(
            create_risk_map(
                combined,
                combined_col,
                "Combined Antarctic Environmental Risk",
                marker_size=5
            ),
            width="stretch",
            key="overview_risk_map"
        )

    else:

        st.info(
            "Combined navigation risk data is not available."
        )

    # Dataset status

    st.markdown(
        '<div class="section-title">📦 Dataset Status</div>',
        unsafe_allow_html=True
    )

    status_rows = [
        {
            "Environmental Layer": "Sea Ice",
            "Status": "Available" if not sea_ice.empty else "Missing",
            "Records": len(sea_ice)
        },
        {
            "Environmental Layer": "Iceberg Risk",
            "Status": "Available" if not iceberg_risk.empty else "Missing",
            "Records": len(iceberg_risk)
        },
        {
            "Environmental Layer": "Iceberg Trajectory",
            "Status": (
                "Available"
                if not iceberg_trajectory.empty
                else "Missing"
            ),
            "Records": len(iceberg_trajectory)
        },
        {
            "Environmental Layer": "Combined Risk",
            "Status": "Available" if not combined.empty else "Missing",
            "Records": len(combined)
        },
        {
            "Environmental Layer": "Risk-Aware Route",
            "Status": (
                "Available"
                if not risk_aware_route.empty
                else "Missing"
            ),
            "Records": len(risk_aware_route)
        },
    ]

    st.dataframe(
        pd.DataFrame(status_rows),
        width="stretch",
        hide_index=True
    )


# ============================================================
# COMBINED RISK
# ============================================================

elif layer == "Combined Risk":

    st.markdown(
        '<div class="section-title">'
        '🌐 Combined Navigation Risk'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'The combined risk surface brings environmental hazards '
        'together into one view for navigation assessment.'
        '</div>',
        unsafe_allow_html=True
    )

    if combined.empty:

        st.error(
            "Combined risk file is missing or could not be loaded."
        )

    else:

        fig = create_risk_map(
            combined,
            combined_col,
            "Antarctic Combined Navigation Risk",
            marker_size=6
        )

        st.plotly_chart(
            fig,
            width="stretch",
            key="combined_navigation_risk_map"
        )

        a, b, c = st.columns(3)

        a.metric(
            "Risk Surface Points",
            f"{len(combined):,}"
        )

        b.metric(
            "Mean Risk",
            f"{average_risk:.2f}"
        )

        c.metric(
            "Peak Risk",
            f"{maximum_risk:.2f}"
        )

        if "combined_risk_level" in combined.columns:

            distribution = (
                combined["combined_risk_level"]
                .value_counts()
                .rename_axis("Risk Level")
                .reset_index(
                    name="Observations"
                )
            )

            st.markdown(
                '<div class="section-title">'
                '📊 Risk Distribution'
                '</div>',
                unsafe_allow_html=True
            )

            fig_dist = px.bar(
                distribution,
                x="Risk Level",
                y="Observations",
                text="Observations"
            )

            fig_dist.update_layout(
                height=400,
                paper_bgcolor="#e8f5fb",
                plot_bgcolor="#ffffff",
                font=dict(
                    color="#173b56"
                ),
                title=dict(
                    text="Navigation Risk Categories",
                    font=dict(
                        color="#0a3552"
                    )
                ),
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            fig_dist.update_traces(
                textposition="auto"
            )

            st.plotly_chart(
                fig_dist,
                width="stretch",
                key="combined_risk_distribution"
            )


# ============================================================
# SEA ICE
# ============================================================

elif layer == "Sea-Ice":

    st.markdown(
        '<div class="section-title">'
        '🧊 Sea-Ice Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Explore sea-ice concentration and the associated '
        'navigation-risk layer.'
        '</div>',
        unsafe_allow_html=True
    )

    if sea_ice.empty:

        st.error(
            "Sea-ice risk file is missing or could not be loaded."
        )

    else:

        fig = create_risk_map(
            sea_ice,
            sea_ice_col,
            "Antarctic Sea-Ice Risk",
            marker_size=6
        )

        st.plotly_chart(
            fig,
            width="stretch",
            key="sea_ice_risk_map"
        )

        predicted_col = first_existing_column(
            sea_ice,
            [
                "predicted_sea_ice_concentration",
                "sea_ice_concentration"
            ]
        )

        a, b, c = st.columns(3)

        a.metric(
            "Grid Cells",
            f"{len(sea_ice):,}"
        )

        if predicted_col:

            sic_values = pd.to_numeric(
                sea_ice[predicted_col],
                errors="coerce"
            ).dropna()

            mean_sic = (
                sic_values.mean()
                if not sic_values.empty
                else np.nan
            )

            b.metric(
                "Mean Sea-Ice Concentration",
                f"{mean_sic:.1f}%"
                if not np.isnan(mean_sic)
                else "N/A"
            )

        else:

            b.metric(
                "Mean Sea-Ice Concentration",
                "N/A"
            )

        if sea_ice_col:

            sea_values = pd.to_numeric(
                sea_ice[sea_ice_col],
                errors="coerce"
            ).dropna()

            max_sic_risk = (
                sea_values.max()
                if not sea_values.empty
                else np.nan
            )

            c.metric(
                "Maximum Ice Risk",
                f"{max_sic_risk:.1f}"
                if not np.isnan(max_sic_risk)
                else "N/A"
            )

        else:

            c.metric(
                "Maximum Ice Risk",
                "N/A"
            )

        preview_cols = [
            col
            for col in [
                "latitude",
                "longitude",
                "sea_ice_concentration",
                "predicted_sea_ice_concentration",
                "sea_ice_risk_score",
                "sea_ice_risk_level"
            ]
            if col in sea_ice.columns
        ]

        if preview_cols:

            st.markdown(
                '<div class="section-title">'
                '📋 Sea-Ice Observations'
                '</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                sea_ice[preview_cols].head(100),
                width="stretch",
                hide_index=True
            )


# ============================================================
# ICEBERGS
# ============================================================

elif layer == "Icebergs":

    st.markdown(
        '<div class="section-title">'
        '🧊 Iceberg Monitoring'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Track iceberg observations, risk levels and predicted movement.'
        '</div>',
        unsafe_allow_html=True
    )

    if iceberg_risk.empty:

        st.error(
            "Iceberg risk file is missing or could not be loaded."
        )

    else:

        fig = create_risk_map(
            iceberg_risk,
            iceberg_col,
            "Antarctic Iceberg Risk",
            marker_size=9
        )

        st.plotly_chart(
            fig,
            width="stretch",
            key="iceberg_risk_map"
        )

        a, b, c = st.columns(3)

        if "iceberg_id" in iceberg_risk.columns:

            a.metric(
                "Tracked Icebergs",
                f"{iceberg_risk['iceberg_id'].nunique():,}"
            )

        else:

            a.metric(
                "Tracked Icebergs",
                "N/A"
            )

        if "prediction_error_km" in iceberg_risk.columns:

            error_values = pd.to_numeric(
                iceberg_risk["prediction_error_km"],
                errors="coerce"
            ).dropna()

            median_error = (
                error_values.median()
                if not error_values.empty
                else np.nan
            )

            b.metric(
                "Median Prediction Error",
                f"{median_error:.2f} km"
                if not np.isnan(median_error)
                else "N/A"
            )

        else:

            b.metric(
                "Median Prediction Error",
                "N/A"
            )

        if iceberg_col:

            iceberg_values = pd.to_numeric(
                iceberg_risk[iceberg_col],
                errors="coerce"
            ).dropna()

            max_iceberg_risk = (
                iceberg_values.max()
                if not iceberg_values.empty
                else np.nan
            )

            c.metric(
                "Maximum Iceberg Risk",
                f"{max_iceberg_risk:.1f}"
                if not np.isnan(max_iceberg_risk)
                else "N/A"
            )

        else:

            c.metric(
                "Maximum Iceberg Risk",
                "N/A"
            )

        # Risk distribution

        if "iceberg_risk_level" in iceberg_risk.columns:

            distribution = (
                iceberg_risk["iceberg_risk_level"]
                .value_counts()
                .rename_axis("Risk Level")
                .reset_index(
                    name="Observations"
                )
            )

            st.markdown(
                '<div class="section-title">'
                '📊 Iceberg Risk Distribution'
                '</div>',
                unsafe_allow_html=True
            )

            fig_dist = px.bar(
                distribution,
                x="Risk Level",
                y="Observations",
                text="Observations"
            )

            fig_dist.update_layout(
                height=400,
                paper_bgcolor="#e8f5fb",
                plot_bgcolor="#ffffff",
                font=dict(
                    color="#173b56"
                ),
                title=dict(
                    text="Iceberg Risk Categories",
                    font=dict(
                        color="#0a3552"
                    )
                )
            )

            st.plotly_chart(
                fig_dist,
                width="stretch",
                key="iceberg_distribution"
            )


# ============================================================
# ROUTES
# ============================================================

elif layer == "Routes":

    st.markdown(
        '<div class="section-title">'
        '🧭 Safer Route Planning'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        f'Compare navigation options for '
        f'<b>{origin}</b> → <b>{destination}</b>.'
        '</div>',
        unsafe_allow_html=True
    )

    st.plotly_chart(
        create_route_map(),
        width="stretch",
        key="route_comparison_map"
    )

    route_rows = []

    for name, route_df in [

        (
            "Shortest",
            shortest_route
        ),

        (
            "Balanced",
            balanced_route
        ),

        (
            "Risk-Aware",
            risk_aware_route
        ),

    ]:

        if route_df.empty:
            continue

        rcol = first_existing_column(
            route_df,
            [
                "combined_risk_score",
                "combined_risk",
                "risk_score"
            ]
        )

        if rcol:

            risks = pd.to_numeric(
                route_df[rcol],
                errors="coerce"
            ).dropna()

            mean_risk = (
                risks.mean()
                if not risks.empty
                else np.nan
            )

            max_risk = (
                risks.max()
                if not risks.empty
                else np.nan
            )

        else:

            mean_risk = np.nan
            max_risk = np.nan

        route_rows.append(
            {
                "Route": name,

                "Distance (km)": (
                    route_distance(route_df)
                ),

                "Mean Risk": mean_risk,

                "Max Risk": max_risk,

                "Waypoints": len(route_df)
            }
        )

    if route_rows:

        st.markdown(
            '<div class="section-title">'
            '📊 Route Comparison'
            '</div>',
            unsafe_allow_html=True
        )

        route_table = pd.DataFrame(
            route_rows
        )

        st.dataframe(
            route_table,
            width="stretch",
            hide_index=True
        )

        # Recommendation

        if (
            "Mean Risk" in route_table.columns
            and route_table["Mean Risk"].notna().any()
        ):

            safest_idx = route_table[
                "Mean Risk"
            ].idxmin()

            safest_route = route_table.loc[
                safest_idx,
                "Route"
            ]

            st.markdown(
                f"""
                <div class="route-card">

                <h3>🧭 Recommended Option</h3>

                <p>
                Based on the available route risk values,
                <b>{safest_route}</b> has the lowest mean
                environmental risk among the displayed routes.
                </p>

                <p>
                Safety preference selected:
                <b>{safety_preference}%</b>
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

    if not route_comparison.empty:

        st.markdown(
            '<div class="section-title">'
            '📁 Stored Optimization Results'
            '</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            route_comparison,
            width="stretch",
            hide_index=True
        )


# ============================================================
# ICEBERG TRAJECTORY EXPLORER
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    '📍 Iceberg Trajectory Explorer'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Select an iceberg to inspect its observed and predicted path.'
    '</div>',
    unsafe_allow_html=True
)


if (
    not iceberg_trajectory.empty
    and "iceberg_id" in iceberg_trajectory.columns
):

    iceberg_ids = sorted(
        iceberg_trajectory["iceberg_id"]
        .dropna()
        .unique()
        .tolist()
    )

    if iceberg_ids:

        selected_iceberg = st.selectbox(
            "Select Iceberg",
            iceberg_ids,
            key="iceberg_selector"
        )

        selected = iceberg_trajectory[
            iceberg_trajectory["iceberg_id"]
            == selected_iceberg
        ].copy()

        if "datetime" in selected.columns:

            selected["datetime"] = pd.to_datetime(
                selected["datetime"],
                errors="coerce"
            )

            selected = selected.sort_values(
                "datetime"
            )

        fig = go.Figure()

        # Observed path

        if {
            "latitude",
            "longitude"
        }.issubset(selected.columns):

            fig.add_trace(
                go.Scattergeo(

                    lon=selected["longitude"],
                    lat=selected["latitude"],

                    mode="lines+markers",

                    line=dict(
                        width=3
                    ),

                    marker=dict(
                        size=5
                    ),

                    name="Observed",

                    hovertemplate=(
                        "Latitude: %{lat:.3f}°<br>"
                        "Longitude: %{lon:.3f}°"
                        "<extra>Observed</extra>"
                    )
                )
            )

        # Predicted path

        if {
            "predicted_latitude",
            "predicted_longitude"
        }.issubset(selected.columns):

            fig.add_trace(
                go.Scattergeo(

                    lon=selected[
                        "predicted_longitude"
                    ],

                    lat=selected[
                        "predicted_latitude"
                    ],

                    mode="lines+markers",

                    line=dict(
                        width=3,
                        dash="dash"
                    ),

                    marker=dict(
                        size=5
                    ),

                    name="Predicted",

                    hovertemplate=(
                        "Latitude: %{lat:.3f}°<br>"
                        "Longitude: %{lon:.3f}°"
                        "<extra>Predicted</extra>"
                    )
                )
            )

        fig.update_geos(
            **polar_geo_settings()
        )

        fig.update_layout(

            title=dict(
                text=(
                    f"Iceberg {selected_iceberg} — "
                    "Observed vs Predicted"
                ),
                font=dict(
                    color="#0a3552"
                )
            ),

            height=550,

            margin=dict(
                l=0,
                r=0,
                t=60,
                b=0
            ),

            paper_bgcolor="#e8f5fb",

            font=dict(
                color="#173b56"
            ),

            legend=dict(
                bgcolor="rgba(255,255,255,0.78)"
            ),

            uirevision="iceberg-trajectory"
        )

        st.plotly_chart(
            fig,
            width="stretch",
            key="iceberg_trajectory_map"
        )

        if "prediction_error_km" in selected.columns:

            error_values = pd.to_numeric(
                selected["prediction_error_km"],
                errors="coerce"
            ).dropna()

            if not error_values.empty:

                c1, c2 = st.columns(2)

                c1.metric(
                    "Mean Prediction Error",
                    f"{error_values.mean():.2f} km"
                )

                c2.metric(
                    "Maximum Prediction Error",
                    f"{error_values.max():.2f} km"
                )

else:

    st.info(
        "Iceberg trajectory data is not available yet."
    )


# ============================================================
# LOCATION ASSESSMENT
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    '📍 Location Assessment'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Enter a location to inspect its available environmental risk.'
    '</div>',
    unsafe_allow_html=True
)


c1, c2, c3 = st.columns(
    [1, 1, 1]
)

with c1:

    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=-48.0,
        value=-65.0,
        step=0.1,
        key="assessment_lat"
    )

with c2:

    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=20.0,
        step=0.1,
        key="assessment_lon"
    )

with c3:

    st.write("")

    st.write("")

    assess = st.button(
        "🔍 Assess Location",
        use_container_width=True
    )


if assess:

    # --------------------------------------------------------
    # Find nearest available combined-risk observation
    # --------------------------------------------------------

    location_risk = None

    nearest_row = None

    if (
        not combined.empty
        and combined_col
        and {
            "latitude",
            "longitude"
        }.issubset(combined.columns)
    ):

        work = combined.copy()

        work["_distance"] = np.sqrt(
            (
                work["latitude"]
                - latitude
            ) ** 2
            +
            (
                work["longitude"]
                - longitude
            ) ** 2
        )

        nearest_idx = work[
            "_distance"
        ].idxmin()

        nearest_row = work.loc[
            nearest_idx
        ]

        location_risk = pd.to_numeric(
            nearest_row[
                combined_col
            ],
            errors="coerce"
        )

    # Fallback only if combined data is unavailable

    if location_risk is None or pd.isna(location_risk):

        location_risk = 0

        st.warning(
            "No valid combined-risk value was found "
            "near this location."
        )

    location_risk = float(
        np.clip(
            location_risk,
            0,
            100
        )
    )

    st.markdown("---")

    st.markdown(
        f"### 📍 {latitude:.2f}°, {longitude:.2f}°"
    )

    if location_risk >= 70:

        st.markdown(
            f"""
            <div class="risk-high">

                <h2>🔴 HIGH RISK</h2>

                <h3>{location_risk:.1f} / 100</h3>

                <p>
                Conditions in the surrounding risk surface
                indicate elevated environmental hazard.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    elif location_risk >= 40:

        st.markdown(
            f"""
            <div class="risk-medium">

                <h2>🟠 MODERATE RISK</h2>

                <h3>{location_risk:.1f} / 100</h3>

                <p>
                Conditions indicate a moderate level
                of environmental concern.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="risk-low">

                <h2>🟢 LOWER RISK</h2>

                <h3>{location_risk:.1f} / 100</h3>

                <p>
                The available combined risk surface
                indicates comparatively lower risk.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    if nearest_row is not None:

        st.markdown(
            '<div class="section-title">'
            'Environmental Details'
            '</div>',
            unsafe_allow_html=True
        )

        detail_cols = [
            col
            for col in [
                combined_col,
                "combined_risk_level",
                "sea_ice_risk_score",
                "iceberg_risk_score",
                "latitude",
                "longitude"
            ]
            if col
            and col in nearest_row.index
        ]

        if detail_cols:

            details = nearest_row[
                detail_cols
            ].to_frame(
                name="Value"
            )

            st.dataframe(
                details,
                width="stretch"
            )


# ============================================================
# DECISION SUMMARY
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    '🎯 POLAR-SAFE Decision Flow'
    '</div>',
    unsafe_allow_html=True
)

a, b, c = st.columns(3)

with a:

    st.markdown(
        """
        <div class="info-card">

        <h3>1️⃣ Understand</h3>

        <p>
        Examine sea-ice and environmental conditions
        across the Antarctic region.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

with b:

    st.markdown(
        """
        <div class="info-card">

        <h3>2️⃣ Track</h3>

        <p>
        Monitor iceberg locations and predicted
        movement to identify potential hazards.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

with c:

    st.markdown(
        """
        <div class="info-card">

        <h3>3️⃣ Navigate</h3>

        <p>
        Compare available routes and prioritize
        lower environmental risk.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DATA PIPELINE
# ============================================================

st.markdown("---")

with st.expander(
    "🔄 How POLAR-SAFE Works"
):

    st.markdown(
        """
        **Environmental Data**

        Sea ice + iceberg observations + trajectory
        information + navigation-risk layers

        ↓

        **Data Processing**

        Cleaning + coordinate alignment +
        risk calculation

        ↓

        **Environmental Risk**

        Individual hazard layers are combined
        into a navigation-risk surface

        ↓

        **Route Analysis**

        Shortest + balanced + risk-aware routes

        ↓

        **Navigation Decision Support**

        Explore the map, assess locations and
        compare safer route options.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    <b>POLAR-SAFE</b> • SIH26059

    <br>

    Antarctic Environmental Risk & Navigation
    Decision-Support Prototype

    <br><br>

    Designed for demonstration and research.
    Final navigation decisions remain with qualified operators.

    </div>
    """,
    unsafe_allow_html=True
)
