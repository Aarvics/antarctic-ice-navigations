import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Antarctica Risk Explorer",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS — BLUE ANTARCTIC THEME
# =========================================================

st.markdown("""
<style>

/* =====================================================
   GLOBAL
   ===================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(147, 197, 253, 0.45),
            transparent 32%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(191, 219, 254, 0.55),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #eaf6ff 0%,
            #dbeafe 45%,
            #eff8ff 100%
        );
    color: #102a43;
}

.main {
    background: transparent;
    color: #102a43;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

/* =====================================================
   TEXT
   ===================================================== */

h1, h2, h3, h4, h5, h6 {
    color: #0b2545 !important;
    font-weight: 700 !important;
}

p, li, label {
    color: #183b56 !important;
}

.stCaption,
.stCaption p {
    color: #486581 !important;
}

/* =====================================================
   HEADER
   ===================================================== */

.hero-title {
    background:
        linear-gradient(
            90deg,
            #082f49,
            #075985,
            #0e7490
        );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.7rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    color: #486581 !important;
    font-size: 1.05rem;
    margin-bottom: 1rem;
}

/* =====================================================
   DIVIDERS
   ===================================================== */

hr {
    border: none;
    height: 1px;
    background: linear-gradient(
        90deg,
        transparent,
        #93c5fd,
        #2563eb,
        #93c5fd,
        transparent
    );
}

/* =====================================================
   SIDEBAR
   ===================================================== */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #dbeafe 0%,
            #e0f2fe 45%,
            #bfdbfe 100%
        );
    border-right: 1px solid rgba(37, 99, 235, 0.20);
}

section[data-testid="stSidebar"] * {
    color: #102a43 !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #082f49 !important;
}

/* =====================================================
   SIDEBAR CHECKBOXES
   ===================================================== */

[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    color: #183b56 !important;
    font-weight: 500;
}

/* =====================================================
   METRIC CARDS
   ===================================================== */

[data-testid="stMetric"] {
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.90),
            rgba(219,234,254,0.88)
        );
    border: 1px solid rgba(59, 130, 246, 0.18);
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow:
        0 8px 24px rgba(30, 64, 175, 0.10);
}

[data-testid="stMetricLabel"] {
    color: #486581 !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #082f49 !important;
    font-weight: 800 !important;
}

[data-testid="stMetricDelta"] {
    color: #075985 !important;
}

/* =====================================================
   GENERAL CARDS
   ===================================================== */

.info-card {
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.95),
            rgba(224,242,254,0.92)
        );
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 18px;
    padding: 22px;
    box-shadow:
        0 10px 30px rgba(30,64,175,0.10);
}

/* =====================================================
   SECTION HEADER
   ===================================================== */

.section-header {
    background:
        linear-gradient(
            90deg,
            rgba(255,255,255,0.75),
            rgba(219,234,254,0.65)
        );
    border-left: 5px solid #2563eb;
    border-radius: 10px;
    padding: 10px 16px;
    margin-top: 10px;
    margin-bottom: 16px;
}

.section-header h3 {
    margin: 0;
    color: #082f49 !important;
}

/* =====================================================
   RISK CARDS
   ===================================================== */

.risk-high {
    background:
        linear-gradient(
            135deg,
            #fee2e2,
            #fecaca
        );
    border: 1px solid #fca5a5;
    color: #7f1d1d !important;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 8px 20px rgba(127,29,29,0.10);
}

.risk-high h2,
.risk-high h3 {
    color: #7f1d1d !important;
}

.risk-medium {
    background:
        linear-gradient(
            135deg,
            #fef3c7,
            #fde68a
        );
    border: 1px solid #fcd34d;
    color: #78350f !important;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 8px 20px rgba(120,53,15,0.10);
}

.risk-medium h2,
.risk-medium h3 {
    color: #78350f !important;
}

.risk-low {
    background:
        linear-gradient(
            135deg,
            #dcfce7,
            #bbf7d0
        );
    border: 1px solid #86efac;
    color: #14532d !important;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 8px 20px rgba(20,83,45,0.10);
}

.risk-low h2,
.risk-low h3 {
    color: #14532d !important;
}

/* =====================================================
   BUTTONS
   ===================================================== */

.stButton > button {
    background:
        linear-gradient(
            135deg,
            #0c4a6e,
            #075985,
            #0369a1
        );
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-weight: 700;
    box-shadow:
        0 5px 15px rgba(3,105,161,0.25);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background:
        linear-gradient(
            135deg,
            #075985,
            #0284c7,
            #0369a1
        );
    color: white !important;
    transform: translateY(-1px);
    box-shadow:
        0 8px 20px rgba(3,105,161,0.30);
}

/* =====================================================
   INPUT BOXES
   ===================================================== */

input {
    color: #102a43 !important;
}

[data-baseweb="input"] {
    background-color: rgba(255,255,255,0.90) !important;
    border-radius: 10px !important;
}

[data-baseweb="select"] {
    background-color: rgba(255,255,255,0.90) !important;
}

[data-baseweb="select"] * {
    color: #102a43 !important;
}

/* =====================================================
   SLIDER
   ===================================================== */

[data-testid="stSlider"] label {
    color: #183b56 !important;
    font-weight: 600;
}

/* =====================================================
   ALERTS
   ===================================================== */

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* =====================================================
   MAP CONTAINER
   ===================================================== */

.map-container {
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.92),
            rgba(219,234,254,0.82)
        );
    border-radius: 18px;
    border: 1px solid rgba(37,99,235,0.16);
    padding: 8px;
    box-shadow:
        0 12px 35px rgba(30,64,175,0.12);
}

/* =====================================================
   FOOTER
   ===================================================== */

.footer {
    text-align: center;
    color: #486581 !important;
    padding: 20px;
    font-size: 0.9rem;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="hero-title">🧊 Antarctica Environmental Risk Explorer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'Explore Antarctic marine, ice and seabed conditions '
    'to identify environmentally sensitive areas.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## 🗺️ Map Layers")

show_risk = st.sidebar.checkbox(
    "Risk",
    True
)

show_sea_ice = st.sidebar.checkbox(
    "Sea Ice Concentration",
    False
)

show_icebergs = st.sidebar.checkbox(
    "Icebergs",
    False
)

show_currents = st.sidebar.checkbox(
    "Ocean Currents",
    False
)

show_bathymetry = st.sidebar.checkbox(
    "Bathymetry",
    False
)

st.sidebar.divider()

st.sidebar.markdown("## ⚙️ Risk Settings")

risk_threshold = st.sidebar.slider(
    "Risk threshold",
    0,
    100,
    60
)


# =========================================================
# TOP STATUS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Region",
        "Antarctica"
    )

with c2:
    st.metric(
        "Overall Risk",
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
        sum([
            show_risk,
            show_sea_ice,
            show_icebergs,
            show_currents,
            show_bathymetry
        ])
    )


# =========================================================
# MAP SECTION
# =========================================================

st.divider()

st.markdown(
    '<div class="section-header"><h3>🗺️ Environmental Risk Map</h3></div>',
    unsafe_allow_html=True
)

# =========================================================
# DEMO DATA
# =========================================================

lat = np.array([
    -65, -66, -67, -68, -69,
    -70, -71, -72, -73, -74
])

lon = np.array([
    -60, -40, -20, 0, 20,
    40, 60, 80, 100, 120
])

risk = np.array([
    35, 48, 61, 72, 82,
    67, 45, 76, 88, 52
])


# =========================================================
# CREATE MAP
# =========================================================

fig = go.Figure()

if show_risk:

    fig.add_trace(
        go.Scattergeo(
            lat=lat,
            lon=lon,
            mode="markers",
            marker=dict(
                size=13,
                color=risk,
                colorscale="RdYlGn_r",
                cmin=0,
                cmax=100,
                line=dict(
                    width=1,
                    color="white"
                ),
                colorbar=dict(
                    title=dict(
                        text="Risk Score",
                        font=dict(
                            color="#102a43"
                        )
                    ),
                    tickfont=dict(
                        color="#102a43"
                    )
                )
            ),
            text=[
                f"Risk Score: {r}"
                for r in risk
            ],
            hovertemplate=
                "<b>%{text}</b><br>" +
                "Latitude: %{lat:.2f}<br>" +
                "Longitude: %{lon:.2f}<extra></extra>"
        )
    )


# =========================================================
# MAP STYLE
# =========================================================

fig.update_geos(
    projection_type="stereographic",
    center=dict(
        lat=-90,
        lon=0
    ),
    projection_rotation=dict(
        lon=0,
        lat=0,
        roll=0
    ),
    showland=True,
    landcolor="#dbeafe",
    showocean=True,
    oceancolor="#e0f2fe",
    showcoastlines=True,
    coastlinecolor="#2563eb",
    coastlinewidth=1.2,
    showcountries=False,
    showlakes=True,
    lakecolor="#bfdbfe",
    bgcolor="#eff8ff"
)

fig.update_layout(
    height=600,
    margin=dict(
        l=0,
        r=0,
        t=0,
        b=0
    ),
    paper_bgcolor="rgba(255,255,255,0.0)",
    plot_bgcolor="rgba(255,255,255,0.0)",
    font=dict(
        color="#102a43"
    )
)

st.markdown(
    '<div class="map-container">',
    unsafe_allow_html=True
)

st.plotly_chart(
    fig,
    use_container_width=True,
    key="environmental_risk_map"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# ENVIRONMENTAL CONDITIONS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-header"><h3>🌍 Environmental Conditions</h3></div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Sea Ice Concentration",
        "82%"
    )

with c2:

    st.metric(
        "Ocean Current",
        "0.74 m/s"
    )

with c3:

    st.metric(
        "Iceberg Activity",
        "HIGH"
    )

with c4:

    st.metric(
        "Bed Depth",
        "-2,431 m"
    )


# =========================================================
# LOCATION ASSESSMENT
# =========================================================

st.divider()

st.markdown(
    '<div class="section-header"><h3>📍 Location Assessment</h3></div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns([1, 1, 1])

with c1:

    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=-48.0,
        value=-65.0,
        step=0.1
    )

with c2:

    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=20.0,
        step=0.1
    )

with c3:

    st.write("")
    st.write("")

    assess = st.button(
        "🔍 Assess Location",
        use_container_width=True
    )


# =========================================================
# LOCATION RESULT
# =========================================================

if assess:

    # Demo calculation
    # Replace with actual integrated data later

    location_risk = 72

    st.divider()

    st.markdown(
        f"### Location: {latitude:.2f}°, {longitude:.2f}°"
    )

    if location_risk >= 70:

        st.markdown(
            f"""
            <div class="risk-high">
                <h2>🔴 HIGH RISK</h2>
                <h3>{location_risk} / 100</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif location_risk >= 40:

        st.markdown(
            f"""
            <div class="risk-medium">
                <h2>🟠 MODERATE RISK</h2>
                <h3>{location_risk} / 100</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="risk-low">
                <h2>🟢 LOW RISK</h2>
                <h3>{location_risk} / 100</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Why?")

    st.write(
        "• High sea-ice concentration"
    )

    st.write(
        "• Elevated iceberg activity"
    )

    st.write(
        "• Strong ocean current"
    )


# =========================================================
# ROUTE PLANNER
# =========================================================

st.divider()

st.markdown(
    '<div class="section-header"><h3>🧭 Safer Route Planner</h3></div>',
    unsafe_allow_html=True
)

st.caption(
    "Find a lower-risk path between two locations."
)

c1, c2 = st.columns(2)

with c1:

    st.markdown("**Start Location**")

    start_lat = st.number_input(
        "Start latitude",
        min_value=-90.0,
        max_value=-48.0,
        value=-65.0,
        key="start_lat"
    )

    start_lon = st.number_input(
        "Start longitude",
        min_value=-180.0,
        max_value=180.0,
        value=0.0,
        key="start_lon"
    )

with c2:

    st.markdown("**Destination**")

    end_lat = st.number_input(
        "Destination latitude",
        min_value=-90.0,
        max_value=-48.0,
        value=-70.0,
        key="end_lat"
    )

    end_lon = st.number_input(
        "Destination longitude",
        min_value=-180.0,
        max_value=180.0,
        value=60.0,
        key="end_lon"
    )


route_button = st.button(
    "🧭 Find Safer Route",
    use_container_width=True
)


if route_button:

    st.success(
        "Safer route calculated successfully."
    )

    st.info(
        "The route planner will use environmental "
        "risk layers to avoid high-risk regions."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    '<div class="footer">'
    '🧊 Antarctica Environmental Risk Explorer '
    '• Integrated marine and cryosphere analysis'
    '</div>',
    unsafe_allow_html=True
)
