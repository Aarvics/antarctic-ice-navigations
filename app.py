
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
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    font-size: 2.4rem;
    font-weight: 700;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

.risk-high {
    background: #fee2e2;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
}

.risk-medium {
    background: #fef3c7;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
}

.risk-low {
    background: #dcfce7;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.title("🧊 Antarctica Environmental Risk Explorer")

st.caption(
    "Explore Antarctic marine, ice and seabed conditions "
    "to identify environmentally sensitive areas."
)

st.divider()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Map Layers")

show_risk = st.sidebar.checkbox("Risk", True)
show_sea_ice = st.sidebar.checkbox("Sea Ice Concentration", False)
show_icebergs = st.sidebar.checkbox("Icebergs", False)
show_currents = st.sidebar.checkbox("Ocean Currents", False)
show_bathymetry = st.sidebar.checkbox("Bathymetry", False)

st.sidebar.divider()

st.sidebar.header("Risk Settings")

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

st.divider()

# =========================================================
# MAP
# =========================================================

st.subheader("🗺️ Environmental Risk Map")

# Demo coordinates
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

fig = go.Figure()

if show_risk:

    fig.add_trace(
        go.Scattergeo(
            lat=lat,
            lon=lon,
            mode="markers",
            marker=dict(
                size=12,
                color=risk,
                colorscale="RdYlGn_r",
                cmin=0,
                cmax=100,
                colorbar=dict(
                    title="Risk"
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

fig.update_geos(
    projection_type="stereographic",
    center=dict(lat=-90, lon=0),
    projection_rotation=dict(lon=0, lat=0, roll=0),
    showland=True,
    showocean=True,
    showcoastlines=True
)
fig.update_layout(
    height=600,
    margin=dict(
        l=0,
        r=0,
        t=0,
        b=0
    ),
    geo=dict(
        bgcolor="white"
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# ENVIRONMENTAL CONDITIONS
# =========================================================

st.subheader("🌍 Environmental Conditions")

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

st.subheader("📍 Location Assessment")

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

st.subheader("🧭 Safer Route Planner")

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

st.caption(
    "Antarctica Environmental Risk Explorer • "
    "Integrated marine and cryosphere analysis"
)
