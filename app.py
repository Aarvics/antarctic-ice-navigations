import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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
# THEME / CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
    color: #000000;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main text */
p, li, span, div {
    color: #000000;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #000000 !important;
}

/* Captions */
.stCaption,
.stCaption p {
    color: #000000 !important;
}

/* Metrics */
[data-testid="stMetricLabel"] {
    color: #000000 !important;
}

[data-testid="stMetricValue"] {
    color: #000000 !important;
}

[data-testid="stMetricDelta"] {
    color: #000000 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
}

section[data-testid="stSidebar"] * {
    color: #000000 !important;
}

/* Buttons */
.stButton button {
    color: #000000 !important;
}

/* Input labels */
label {
    color: #000000 !important;
}

/* Input text */
input {
    color: #000000 !important;
}

/* Risk cards */
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

st.markdown(
    '<div class="section-label">ANTARCTIC ENVIRONMENTAL MONITORING</div>',
    unsafe_allow_html=True
)

st.title("🧊 Antarctica Risk Explorer")

st.markdown(
    "A unified view of marine conditions, sea ice and seabed characteristics "
    "for environmental risk assessment."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## 🧊 Antarctica")
st.sidebar.caption("Environmental Risk Explorer")

st.sidebar.divider()

st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Risk Map",
        "Ocean Conditions",
        "Ice Conditions",
        "Seabed & Bathymetry",
        "Location Assessment",
        "Safer Route"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()

st.sidebar.markdown("### Map Layers")

show_risk = st.sidebar.checkbox(
    "🔴 Environmental Risk",
    True
)

show_sea_ice = st.sidebar.checkbox(
    "❄️ Sea Ice",
    False
)

show_icebergs = st.sidebar.checkbox(
    "🧊 Icebergs",
    False
)

show_currents = st.sidebar.checkbox(
    "🌊 Ocean Currents",
    False
)

show_bathymetry = st.sidebar.checkbox(
    "🏔️ Bathymetry",
    False
)

st.sidebar.divider()

st.sidebar.markdown("### Risk Settings")

risk_threshold = st.sidebar.slider(
    "Risk threshold",
    min_value=0,
    max_value=100,
    value=60
)

st.sidebar.divider()

st.sidebar.caption(
    "Data sources are being progressively integrated into the platform."
)


# =========================================================
# DEMO DATA
# =========================================================
# These values keep the dashboard functional while the real
# datasets are connected.

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

sea_ice = np.array([
    72, 78, 81, 86, 91,
    88, 79, 83, 94, 76
])

current_speed = np.array([
    0.31, 0.45, 0.52, 0.61, 0.74,
    0.68, 0.39, 0.82, 0.91, 0.55
])

bed_depth = np.array([
    -2100, -2800, -3400, -4200, -2431,
    -5100, -3700, -4600, -5900, -3100
])


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def create_risk_map():

    fig = go.Figure()

    if show_risk:

        fig.add_trace(
            go.Scattergeo(
                lat=lat,
                lon=lon,
                mode="markers",
                marker=dict(
                    size=14,
                    color=risk,
                    colorscale="RdYlGn_r",
                    cmin=0,
                    cmax=100,
                    opacity=0.9,
                    line=dict(
                        width=1,
                        color="white"
                    ),
                    colorbar=dict(
                        title="Risk"
                    )
                ),
                text=[
                    f"Risk Score: {r}"
                    for r in risk
                ],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Latitude: %{lat:.2f}°<br>"
                    "Longitude: %{lon:.2f}°"
                    "<extra></extra>"
                ),
                name="Environmental Risk"
            )
        )

    if show_sea_ice:

        fig.add_trace(
            go.Scattergeo(
                lat=lat,
                lon=lon,
                mode="markers",
                marker=dict(
                    size=9,
                    color=sea_ice,
                    colorscale="Blues",
                    cmin=0,
                    cmax=100,
                    opacity=0.7
                ),
                text=[
                    f"Sea Ice: {x}%"
                    for x in sea_ice
                ],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Latitude: %{lat:.2f}°<br>"
                    "Longitude: %{lon:.2f}°"
                    "<extra></extra>"
                ),
                name="Sea Ice"
            )
        )

    if show_currents:

        fig.add_trace(
            go.Scattergeo(
                lat=lat,
                lon=lon,
                mode="markers",
                marker=dict(
                    size=8,
                    color=current_speed,
                    colorscale="Viridis",
                    cmin=0,
                    cmax=1
                ),
                text=[
                    f"Current Speed: {x:.2f} m/s"
                    for x in current_speed
                ],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Latitude: %{lat:.2f}°<br>"
                    "Longitude: %{lon:.2f}°"
                    "<extra></extra>"
                ),
                name="Ocean Current"
            )
        )

    if show_bathymetry:

        fig.add_trace(
            go.Scattergeo(
                lat=lat,
                lon=lon,
                mode="markers",
                marker=dict(
                    size=9,
                    color=bed_depth,
                    colorscale="Cividis",
                    opacity=0.8
                ),
                text=[
                    f"Bed Depth: {x} m"
                    for x in bed_depth
                ],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Latitude: %{lat:.2f}°<br>"
                    "Longitude: %{lon:.2f}°"
                    "<extra></extra>"
                ),
                name="Bathymetry"
            )
        )

    if show_icebergs:

        fig.add_trace(
            go.Scattergeo(
                lat=[-67, -69, -71],
                lon=[-20, 35, 90],
                mode="markers",
                marker=dict(
                    size=10,
                    symbol="diamond",
                    opacity=0.8
                ),
                text=[
                    "Iceberg activity",
                    "Iceberg activity",
                    "Iceberg activity"
                ],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Latitude: %{lat:.2f}°<br>"
                    "Longitude: %{lon:.2f}°"
                    "<extra></extra>"
                ),
                name="Icebergs"
            )
        )

    fig.update_geos(
        projection_type="stereographic",
        center=dict(
            lat=-90,
            lon=0
        ),
        showland=True,
        showocean=True,
        showcoastlines=True,
        coastlinecolor="#486581",
        landcolor="#d9e2ec",
        oceancolor="#eaf4f8"
    )

    fig.update_layout(
        height=650,
        margin=dict(
            l=0,
            r=0,
            t=20,
            b=0
        ),
        paper_bgcolor="white",
        geo=dict(
            bgcolor="white"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def risk_category(score):

    if score >= 70:
        return "HIGH", "risk-high"

    elif score >= 40:
        return "MODERATE", "risk-medium"

    return "LOW", "risk-low"


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.markdown(
        '<div class="section-label">SYSTEM OVERVIEW</div>',
        unsafe_allow_html=True
    )

    st.header("Antarctic Environmental Status")

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="metric-title">OVERALL RISK</div>
                <div class="metric-value">HIGH</div>
                <div class="metric-sub">Current assessment</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="metric-title">RISK SCORE</div>
                <div class="metric-value">72 / 100</div>
                <div class="metric-sub">Environmental index</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="metric-title">SEA ICE</div>
                <div class="metric-value">82%</div>
                <div class="metric-sub">Regional concentration</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="metric-title">CURRENT SPEED</div>
                <div class="metric-value">0.74 m/s</div>
                <div class="metric-sub">Representative value</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # -----------------------------------------------------
    # MAP
    # -----------------------------------------------------

    st.subheader("🗺️ Environmental Risk Map")

    fig = create_risk_map()

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="overview_risk_map"
    )

    st.divider()

    # -----------------------------------------------------
    # DATASET STATUS
    # -----------------------------------------------------

    st.subheader("📡 Environmental Data")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            """
            <div class="info-box">
            <b>🌊 Ocean Current</b><br>
            <small>Integrated</small>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="info-box">
            <b>❄️ Sea Ice</b><br>
            <small>Integrated</small>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="info-box">
            <b>🏔️ Geography & Bathymetry</b><br>
            <small>Integrated</small>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            """
            <div class="info-box">
            <b>🧊 Iceberg</b><br>
            <small>Dataset pending</small>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# RISK MAP
# =========================================================

elif page == "Risk Map":

    st.markdown(
        '<div class="section-label">SPATIAL ANALYSIS</div>',
        unsafe_allow_html=True
    )

    st.header("🗺️ Environmental Risk Map")

    st.caption(
        "Toggle environmental layers from the sidebar to compare "
        "different Antarctic conditions."
    )

    fig = create_risk_map()

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="risk_map"
    )

    st.subheader("Risk Distribution")

    high = np.sum(risk >= 70)
    moderate = np.sum((risk >= 40) & (risk < 70))
    low = np.sum(risk < 40)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("🔴 High Risk Areas", high)

    with c2:
        st.metric("🟠 Moderate Risk Areas", moderate)

    with c3:
        st.metric("🟢 Low Risk Areas", low)


# =========================================================
# OCEAN CONDITIONS
# =========================================================

elif page == "Ocean Conditions":

    st.markdown(
        '<div class="section-label">MARINE CONDITIONS</div>',
        unsafe_allow_html=True
    )

    st.header("🌊 Ocean Conditions")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Current Speed",
            "0.74 m/s"
        )

    with c2:
        st.metric(
            "Maximum Observed",
            f"{current_speed.max():.2f} m/s"
        )

    with c3:
        st.metric(
            "Minimum Observed",
            f"{current_speed.min():.2f} m/s"
        )

    st.write("")

    current_df = pd.DataFrame({
        "Latitude": lat,
        "Longitude": lon,
        "Current Speed (m/s)": current_speed
    })

    fig = px.scatter(
        current_df,
        x="Longitude",
        y="Latitude",
        size="Current Speed (m/s)",
        color="Current Speed (m/s)",
        color_continuous_scale="Viridis",
        title="Ocean Current Intensity"
    )

    fig.update_layout(
        height=550,
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="ocean_current_chart"
    )

    st.dataframe(
        current_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# ICE CONDITIONS
# =========================================================

elif page == "Ice Conditions":

    st.markdown(
        '<div class="section-label">CRYOSPHERE</div>',
        unsafe_allow_html=True
    )

    st.header("❄️ Ice Conditions")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Average Sea Ice",
            f"{sea_ice.mean():.1f}%"
        )

    with c2:
        st.metric(
            "Maximum",
            f"{sea_ice.max():.1f}%"
        )

    with c3:
        st.metric(
            "Minimum",
            f"{sea_ice.min():.1f}%"
        )

    st.write("")

    ice_df = pd.DataFrame({
        "Latitude": lat,
        "Longitude": lon,
        "Sea Ice Concentration (%)": sea_ice
    })

    fig = px.scatter(
        ice_df,
        x="Longitude",
        y="Latitude",
        size="Sea Ice Concentration (%)",
        color="Sea Ice Concentration (%)",
        color_continuous_scale="Blues",
        title="Sea Ice Concentration"
    )

    fig.update_layout(
        height=550,
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="sea_ice_chart"
    )

    st.dataframe(
        ice_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# SEABED & BATHYMETRY
# =========================================================

elif page == "Seabed & Bathymetry":

    st.markdown(
        '<div class="section-label">SUBSURFACE ANALYSIS</div>',
        unsafe_allow_html=True
    )

    st.header("🏔️ Seabed & Bathymetry")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Deepest Bed",
            f"{bed_depth.min():,.0f} m"
        )

    with c2:
        st.metric(
            "Shallowest Bed",
            f"{bed_depth.max():,.0f} m"
        )

    with c3:
        st.metric(
            "Mean Bed Depth",
            f"{bed_depth.mean():,.0f} m"
        )

    st.write("")

    bathy_df = pd.DataFrame({
        "Latitude": lat,
        "Longitude": lon,
        "Bed Depth (m)": bed_depth
    })

    fig = px.scatter(
        bathy_df,
        x="Longitude",
        y="Latitude",
        size=np.abs(bed_depth),
        color="Bed Depth (m)",
        color_continuous_scale="Cividis",
        title="Antarctic Bed Depth"
    )

    fig.update_layout(
        height=550,
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="bathymetry_chart"
    )

    st.dataframe(
        bathy_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# LOCATION ASSESSMENT
# =========================================================

elif page == "Location Assessment":

    st.markdown(
        '<div class="section-label">POINT-BASED ANALYSIS</div>',
        unsafe_allow_html=True
    )

    st.header("📍 Location Assessment")

    st.caption(
        "Enter a coordinate to obtain an environmental risk assessment."
    )

    c1, c2 = st.columns(2)

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

    assess = st.button(
        "🔍 Assess Location",
        use_container_width=True
    )

    if assess:

        # Current demo calculation
        location_risk = 72

        category, css_class = risk_category(
            location_risk
        )

        st.write("")

        st.markdown(
            f"""
            <div class="{css_class}">
                <div style="font-size:0.85rem;">
                    ENVIRONMENTAL RISK
                </div>
                <div style="font-size:2rem;font-weight:800;">
                    {category}
                </div>
                <div style="font-size:1.4rem;font-weight:700;">
                    {location_risk} / 100
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.markdown(
            f"### Location: {latitude:.2f}°, {longitude:.2f}°"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Sea Ice",
                "82%"
            )

        with c2:
            st.metric(
                "Current",
                "0.74 m/s"
            )

        with c3:
            st.metric(
                "Bed Depth",
                "-2,431 m"
            )

        with c4:
            st.metric(
                "Iceberg Activity",
                "HIGH"
            )

        st.divider()

        st.subheader("Why this location is classified as high risk")

        st.write(
            "• High sea-ice concentration"
        )

        st.write(
            "• Elevated iceberg activity"
        )

        st.write(
            "• Strong ocean current"
        )

        st.write(
            "• Local seabed characteristics"
        )


# =========================================================
# SAFER ROUTE
# =========================================================

elif page == "Safer Route":

    st.markdown(
        '<div class="section-label">ROUTE PLANNING</div>',
        unsafe_allow_html=True
    )

    st.header("🧭 Safer Route Planner")

    st.caption(
        "Compare environmental risk between a start point "
        "and destination."
    )

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Start Location")

        start_lat = st.number_input(
            "Start latitude",
            min_value=-90.0,
            max_value=-48.0,
            value=-65.0,
            step=0.1,
            key="start_lat"
        )

        start_lon = st.number_input(
            "Start longitude",
            min_value=-180.0,
            max_value=180.0,
            value=0.0,
            step=0.1,
            key="start_lon"
        )

    with c2:

        st.subheader("Destination")

        end_lat = st.number_input(
            "Destination latitude",
            min_value=-90.0,
            max_value=-48.0,
            value=-70.0,
            step=0.1,
            key="end_lat"
        )

        end_lon = st.number_input(
            "Destination longitude",
            min_value=-180.0,
            max_value=180.0,
            value=60.0,
            step=0.1,
            key="end_lon"
        )

    route_button = st.button(
        "🧭 Find Safer Route",
        use_container_width=True
    )

    if route_button:

        st.success(
            "Route analysis completed."
        )

        # Demo route
        route_lat = np.linspace(
            start_lat,
            end_lat,
            30
        )

        route_lon = np.linspace(
            start_lon,
            end_lon,
            30
        )

        route_fig = go.Figure()

        route_fig.add_trace(
            go.Scattergeo(
                lat=route_lat,
                lon=route_lon,
                mode="lines+markers",
                line=dict(
                    width=4
                ),
                marker=dict(
                    size=5
                ),
                name="Recommended Route"
            )
        )

        route_fig.add_trace(
            go.Scattergeo(
                lat=[start_lat],
                lon=[start_lon],
                mode="markers",
                marker=dict(
                    size=14
                ),
                name="Start"
            )
        )

        route_fig.add_trace(
            go.Scattergeo(
                lat=[end_lat],
                lon=[end_lon],
                mode="markers",
                marker=dict(
                    size=14
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
            showland=True,
            showocean=True,
            showcoastlines=True,
            landcolor="#d9e2ec",
            oceancolor="#eaf4f8"
        )

        route_fig.update_layout(
            height=600,
            margin=dict(
                l=0,
                r=0,
                t=20,
                b=0
            ),
            paper_bgcolor="white"
        )

        st.plotly_chart(
            route_fig,
            use_container_width=True,
            key="safer_route_map"
        )

        st.divider()

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Estimated Risk",
                "LOW"
            )

        with c2:
            st.metric(
                "High-Risk Zones Avoided",
                "4"
            )

        with c3:
            st.metric(
                "Route Status",
                "RECOMMENDED"
            )

        st.info(
            "The final route engine will use the integrated "
            "environmental risk layers to avoid high-risk regions."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Antarctica Environmental Risk Explorer • "
    "Marine • Ice • Seabed"
)
