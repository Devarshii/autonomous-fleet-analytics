import streamlit as st
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Autonomous Fleet Analytics",
    page_icon="🚕",
    layout="wide"
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🚕 Fleet Control Center")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Demand Forecasting",
        "Fleet Positioning",
        "Simulation Results",
        "Maintenance Analytics",
        "Stuck-Risk Monitoring",
        "Shock & Drift Analysis"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Autonomous Fleet Analytics | NYC Pilot"
)


# ============================================================
# LOAD PROJECT DATA
# ============================================================

@st.cache_data
def load_project_data():

    demand = pd.read_parquet(
        "data/processed/nyc_hourly_demand_complete.parquet"
    )

    forecast = pd.read_parquet(
        "data/processed/lightgbm_predictions.parquet"
    )

    maintenance = pd.read_csv(
        "outputs/maintenance/fleet_health_status.csv"
    )

    stuck_risk = pd.read_csv(
        "outputs/stuck_risk/stuck_risk_analysis.csv"
    )

    drift = pd.read_csv(
        "outputs/shock_drift/drift_summary.csv"
    )

    return demand, forecast, maintenance, stuck_risk, drift


demand_df, forecast_df, maintenance_df, stuck_df, drift_df = (
    load_project_data()
)


# ============================================================
# GLOBAL HEADER
# ============================================================

st.title("🚕 Autonomous Fleet Analytics")

st.subheader(
    "Demand-Aware Fleet Positioning and Health Analytics "
    "for Autonomous Vehicles"
)

st.markdown(
    """
    An end-to-end analytics platform for autonomous fleet operations,
    combining demand forecasting, vehicle positioning, simulation,
    predictive maintenance, stuck-risk monitoring, and demand-drift
    detection.
    """
)

st.divider()

st.caption(
    f"Current module: **{page}**"
)


# ============================================================
# EXECUTIVE OVERVIEW PAGE
# ============================================================

if page == "Executive Overview":

    st.header("📊 Executive Fleet Overview")

    total_demand = int(
        demand_df["demand"].sum()
    )

    avg_hourly_demand = (
        demand_df
        .groupby("pickup_hour")["demand"]
        .sum()
        .mean()
    )

    avg_maintenance_risk = maintenance_df[
        "maintenance_risk_score"
    ].mean()

    avg_stuck_risk = stuck_df[
        "stuck_risk_score"
    ].mean()

    # --------------------------------------------------------
    # EXECUTIVE KPIs
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "January Ride Demand",
            f"{total_demand:,}"
        )

    with col2:
        st.metric(
            "Average Hourly Demand",
            f"{avg_hourly_demand:,.0f}"
        )

    with col3:
        st.metric(
            "Average Maintenance Risk",
            f"{avg_maintenance_risk:.1f}/100"
        )

    with col4:
        st.metric(
            "Average Stuck Risk",
            f"{avg_stuck_risk:.1f}/100"
        )

    # --------------------------------------------------------
    # HOURLY DEMAND
    # --------------------------------------------------------

    st.subheader("📈 NYC Ride Demand by Hour")

    hourly_demand = (
        demand_df
        .groupby("hour")["demand"]
        .sum()
        .reset_index()
    )

    st.line_chart(
        hourly_demand,
        x="hour",
        y="demand"
    )

    peak_hour = hourly_demand.loc[
        hourly_demand["demand"].idxmax()
    ]

    st.info(
        f"🚕 Peak demand occurs around "
        f"**{int(peak_hour['hour']):02d}:00**, with "
        f"**{int(peak_hour['demand']):,} rides** recorded "
        f"during January."
    )

    # --------------------------------------------------------
    # EXECUTIVE SUMMARY
    # --------------------------------------------------------

    st.subheader("Executive Summary")

    st.success(
        """
        The analytics platform combines forecasting and operational
        decision support to help an autonomous mobility operator
        position vehicles more effectively, reduce service mismatch,
        prioritize maintenance, identify risky operating conditions,
        and respond to changing demand patterns.
        """
    )


# ============================================================
# DEMAND FORECASTING PAGE
# ============================================================

if page == "Demand Forecasting":

    st.header("🔮 Demand Forecasting")

    st.markdown(
        """
        The forecasting module predicts hourly ride demand across
        NYC taxi zones using historical demand patterns and
        time-based features.
        """
    )

    # --------------------------------------------------------
    # MODEL DATASET SUMMARY
    # --------------------------------------------------------

    st.subheader("Model Dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Prediction Records",
            f"{len(forecast_df):,}"
        )

    with col2:
        st.metric(
            "Average Actual Demand",
            f"{forecast_df['demand'].mean():.2f}"
        )

    with col3:
        st.metric(
            "Maximum Actual Demand",
            f"{forecast_df['demand'].max():,.0f}"
        )

    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    st.subheader("Forecast Data Preview")

    st.dataframe(
        forecast_df.head(20),
        width="stretch"
    )

    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    st.subheader("Actual vs Predicted Demand")

    st.image(
        "outputs/forecasting/actual_vs_predicted_demand.png",
        caption=(
            "LightGBM forecast compared with actual NYC ride demand"
        ),
        width="stretch"
    )

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader("Baseline vs LightGBM Performance")

    st.image(
        "outputs/forecasting/baseline_vs_lightgbm_mae.png",
        caption=(
            "Historical baseline vs LightGBM — "
            "Mean Absolute Error"
        ),
        width="stretch"
    )

    st.success(
        """
        🚀 **LightGBM improved forecasting accuracy.**

        The model reduced Mean Absolute Error from
        **4.40 to 3.53 rides per zone-hour**, representing
        approximately **19.9% lower forecasting error**
        than the historical baseline.
        """
    )


# ============================================================
# FLEET POSITIONING PAGE
# ============================================================

if page == "Fleet Positioning":

    st.header("🚕 Fleet Positioning")

    st.markdown(
        """
        This module recommends where idle autonomous vehicles
        should be positioned using predicted ride demand across
        NYC taxi zones.
        """
    )

    # --------------------------------------------------------
    # POSITIONING OVERVIEW
    # --------------------------------------------------------

    st.subheader("Demand-Aware Positioning Strategy")

    st.markdown(
        """
        Available vehicles are allocated in proportion to expected
        demand. Higher-demand zones receive more vehicles so the
        fleet can reduce demand-supply mismatch and improve service
        availability.
        """
    )

    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

    st.subheader("Recommended Fleet Positioning Map")

    st.image(
        "outputs/fleet_positioning/fleet_positioning_map.png",
        caption=(
            "Demand-aware recommended autonomous vehicle "
            "positioning across NYC"
        ),
        width="stretch"
    )

    # --------------------------------------------------------
    # TOP ALLOCATION ZONES
    # --------------------------------------------------------

    st.subheader("Recommended Vehicle Allocation")

    st.image(
        "outputs/fleet_positioning/recommended_vehicle_allocation.png",
        caption=(
            "Top NYC taxi zones by recommended autonomous "
            "vehicle allocation"
        ),
        width="stretch"
    )


# ============================================================
# SIMULATION RESULTS PAGE
# ============================================================

if page == "Simulation Results":

    simulation_summary = pd.read_csv(
        "outputs/simulation/multi_hour_simulation_summary.csv"
    )

    sim = simulation_summary.iloc[0]

    st.header("📊 Simulation Results")

    st.markdown(
        """
        A 7-day simulation evaluates how forecast-driven vehicle
        repositioning performs relative to a baseline positioning
        strategy.
        """
    )

    # --------------------------------------------------------
    # SIMULATION KPIs
    # --------------------------------------------------------

    st.subheader("Simulation Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Hours Simulated",
            f"{int(sim['hours_simulated'])}"
        )

    with col2:
        st.metric(
            "Total Ride Demand",
            f"{int(sim['total_demand']):,}"
        )

    with col3:
        st.metric(
            "Fleet Size",
            f"{int(sim['fleet_size_per_hour']):,}"
        )

    with col4:
        st.metric(
            "Forecast Strategy Wins",
            f"{int(sim['policy_b_better_hours'])} hrs"
        )

    # --------------------------------------------------------
    # ALLOCATION ERROR
    # --------------------------------------------------------

    st.subheader("Positioning Strategy Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Baseline Allocation Error",
            f"{sim['policy_a_avg_allocation_mae']:.2f}"
        )

    with col2:
        st.metric(
            "Forecast-Driven Allocation Error",
            f"{sim['policy_b_avg_allocation_mae']:.2f}",
            delta=(
                f"-{sim['allocation_mae_improvement_pct']:.1f}% error"
            )
        )

    # --------------------------------------------------------
    # RIDES SERVED
    # --------------------------------------------------------

    st.subheader("Ride Service Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Baseline Rides Served",
            f"{int(sim['policy_a_rides_served']):,}"
        )

    with col2:
        st.metric(
            "Forecast Strategy Rides Served",
            f"{int(sim['policy_b_rides_served']):,}"
        )

    with col3:
        st.metric(
            "Additional Rides Served",
            f"+{int(sim['additional_rides_served']):,}"
        )

    # --------------------------------------------------------
    # SERVICE RATE
    # --------------------------------------------------------

    st.subheader("Service Rate Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Baseline Service Rate",
            f"{sim['policy_a_service_rate']:.2f}%"
        )

    with col2:
        st.metric(
            "Forecast Strategy Service Rate",
            f"{sim['policy_b_service_rate']:.2f}%"
        )

    with col3:
        st.metric(
            "Service Rate Improvement",
            f"+{sim['service_rate_improvement_pp']:.2f} pp"
        )

    # --------------------------------------------------------
    # BUSINESS IMPACT
    # --------------------------------------------------------

    st.subheader("Business Impact")

    st.success(
        f"""
        🚀 **Forecast-driven fleet positioning outperformed the baseline strategy.**

        - Allocation error reduced by **{sim['allocation_mae_improvement_pct']:.1f}%**
        - Forecast-driven positioning performed better in
          **{int(sim['policy_b_better_hours'])} of {int(sim['hours_simulated'])} hours**
        - **{int(sim['additional_rides_served']):,} additional rides** were served
        - Service rate improved by
          **{sim['service_rate_improvement_pp']:.2f} percentage points**

        The simulation demonstrates measurable operational value
        from combining demand forecasting with vehicle repositioning.
        """
    )


# ============================================================
# MAINTENANCE ANALYTICS PAGE
# ============================================================

if page == "Maintenance Analytics":

    st.header("🔧 Maintenance Analytics")

    st.markdown(
        """
        Simulated autonomous vehicle health signals are used to
        prioritize preventive maintenance and identify vehicles
        that should temporarily be removed from service.
        """
    )

    # --------------------------------------------------------
    # FLEET HEALTH
    # --------------------------------------------------------

    st.subheader("Fleet Health Overview")

    total_vehicles = len(maintenance_df)

    available_vehicles = (
        maintenance_df["operational_status"] == "AVAILABLE"
    ).sum()

    maintenance_vehicles = (
        maintenance_df["operational_status"] == "MAINTENANCE"
    ).sum()

    urgent_vehicles = (
        maintenance_df["operational_status"]
        == "URGENT_MAINTENANCE"
    ).sum()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Fleet",
            f"{total_vehicles:,}"
        )

    with col2:
        st.metric(
            "Available Vehicles",
            f"{available_vehicles:,}"
        )

    with col3:
        st.metric(
            "Maintenance Required",
            f"{maintenance_vehicles:,}"
        )

    with col4:
        st.metric(
            "Urgent Maintenance",
            f"{urgent_vehicles:,}"
        )

    # --------------------------------------------------------
    # RISK SUMMARY
    # --------------------------------------------------------

    st.subheader("Maintenance Risk Summary")

    avg_risk = maintenance_df[
        "maintenance_risk_score"
    ].mean()

    max_risk = maintenance_df[
        "maintenance_risk_score"
    ].max()

    high_risk_count = (
        maintenance_df["maintenance_risk_score"] >= 70
    ).sum()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Risk",
            f"{avg_risk:.1f}/100"
        )

    with col2:
        st.metric(
            "Highest Risk",
            f"{max_risk:.1f}/100"
        )

    with col3:
        st.metric(
            "High-Risk Vehicles",
            f"{high_risk_count:,}"
        )

    # --------------------------------------------------------
    # TOP VEHICLES
    # --------------------------------------------------------

    st.subheader("Highest-Priority Maintenance Vehicles")

    priority_vehicles = (
        maintenance_df
        .sort_values(
            "maintenance_risk_score",
            ascending=False
        )
        .head(15)
    )

    st.dataframe(
        priority_vehicles[
            [
                "vehicle_id",
                "maintenance_risk_score",
                "maintenance_priority",
                "operational_status"
            ]
        ],
        width="stretch"
    )

    # --------------------------------------------------------
    # SCHEDULING
    # --------------------------------------------------------

    st.subheader("Scheduled Maintenance Windows")

    maintenance_schedule = pd.read_csv(
        "outputs/maintenance/maintenance_schedule.csv"
    )

    st.markdown(
        """
        Maintenance is scheduled during lower-demand periods
        whenever possible to protect revenue-generating fleet
        capacity during peak operating hours.
        """
    )

    scheduled_count = len(
        maintenance_schedule
    )

    urgent_scheduled = (
        maintenance_schedule["operational_status"]
        == "URGENT_MAINTENANCE"
    ).sum()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Vehicles Scheduled",
            f"{scheduled_count:,}"
        )

    with col2:
        st.metric(
            "Urgent Vehicles Scheduled",
            f"{urgent_scheduled:,}"
        )

    st.dataframe(
        maintenance_schedule.head(20),
        width="stretch"
    )

    st.info(
        """
        Vehicle health data in this project is simulated for
        analytical demonstration and does not represent
        production autonomous-vehicle telemetry.
        """
    )


# ============================================================
# STUCK-RISK MONITORING PAGE
# ============================================================

if page == "Stuck-Risk Monitoring":

    st.header("⚠️ Stuck-Risk Monitoring")

    st.markdown(
        """
        This module identifies NYC zone-hour combinations where
        autonomous vehicles may face elevated operational risk
        and require intervention.
        """
    )

    # --------------------------------------------------------
    # OVERVIEW KPIs
    # --------------------------------------------------------

    st.subheader("Stuck-Risk Overview")

    average_stuck_risk = stuck_df[
        "stuck_risk_score"
    ].mean()

    maximum_stuck_risk = stuck_df[
        "stuck_risk_score"
    ].max()

    critical_records = (
        stuck_df["stuck_risk_score"] >= 80
    ).sum()

    avoid_idle_count = (
        stuck_df["av_recommendation"]
        == "AVOID_IDLE_POSITIONING"
    ).sum()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Average Stuck Risk",
            f"{average_stuck_risk:.1f}/100"
        )

        st.metric(
            "Critical Risk Records",
            f"{critical_records:,}"
        )

    with col2:
        st.metric(
            "Highest Stuck Risk",
            f"{maximum_stuck_risk:.1f}/100"
        )

        st.metric(
            "Avoid-Idle Recommendations",
            f"{avoid_idle_count:,}"
        )

    # --------------------------------------------------------
    # HIGH-RISK CONDITIONS
    # --------------------------------------------------------

    st.subheader("Highest-Risk Zone-Hour Conditions")

    highest_risk = (
        stuck_df
        .sort_values(
            "stuck_risk_score",
            ascending=False
        )
        .head(20)
    )

    display_columns = [
        "pickup_hour",
        "LocationID",
        "Borough",
        "Zone",
        "demand",
        "stuck_risk_score",
        "av_recommendation"
    ]

    available_display_columns = [
        column
        for column in display_columns
        if column in highest_risk.columns
    ]

    st.dataframe(
        highest_risk[
            available_display_columns
        ],
        width="stretch"
    )

    # --------------------------------------------------------
    # ZONE-LEVEL SUMMARY
    # --------------------------------------------------------

    st.subheader("Highest Average Stuck-Risk Zones")

    zone_risk = (
        stuck_df
        .groupby(
            ["LocationID", "Borough", "Zone"],
            as_index=False
        )
        .agg(
            average_stuck_risk=(
                "stuck_risk_score",
                "mean"
            ),
            maximum_stuck_risk=(
                "stuck_risk_score",
                "max"
            ),
            average_demand=(
                "demand",
                "mean"
            )
        )
        .sort_values(
            "average_stuck_risk",
            ascending=False
        )
        .head(15)
    )

    st.dataframe(
        zone_risk,
        width="stretch"
    )

    # --------------------------------------------------------
    # OPERATIONAL GUIDANCE
    # --------------------------------------------------------

    st.subheader("AV Operational Guidance")

    avoid_positioning = (
        stuck_df[
            stuck_df["av_recommendation"]
            == "AVOID_IDLE_POSITIONING"
        ]
        .sort_values(
            "stuck_risk_score",
            ascending=False
        )
        .head(15)
    )

    if not avoid_positioning.empty:

        st.warning(
            """
            ⚠️ High-risk zone-hour combinations should not be
            preferred for idle vehicle staging.

            Vehicles can instead be repositioned toward nearby
            lower-risk zones while maintaining access to
            forecasted demand.
            """
        )

        guidance_columns = [
            "pickup_hour",
            "Borough",
            "Zone",
            "demand",
            "stuck_risk_score",
            "av_recommendation"
        ]

        available_guidance_columns = [
            column
            for column in guidance_columns
            if column in avoid_positioning.columns
        ]

        st.dataframe(
            avoid_positioning[
                available_guidance_columns
            ],
            width="stretch"
        )

    else:

        st.success(
            "✅ No avoid-idle recommendations were found."
        )

    st.info(
        """
        Stuck-risk signals are analytical proxies designed for
        this project and should not be interpreted as direct
        production AV telemetry.
        """
    )


# ============================================================
# SHOCK & DRIFT ANALYSIS PAGE
# ============================================================

if page == "Shock & Drift Analysis":

    st.header("🌨️ Shock & Drift Analysis")

    st.markdown(
        """
        This module monitors changing ride-demand behavior and
        evaluates how abnormal events may affect forecasting,
        fleet positioning, and model reliability.
        """
    )

    # --------------------------------------------------------
    # DRIFT VALUES
    # --------------------------------------------------------

    drift_lookup = dict(
        zip(
            drift_df["metric"],
            drift_df["value"]
        )
    )

    reference_average = float(
        drift_lookup.get(
            "reference_average_demand",
            0
        )
    )

    current_average = float(
        drift_lookup.get(
            "current_average_demand",
            0
        )
    )

    demand_change = float(
        drift_lookup.get(
            "demand_change_percent",
            0
        )
    )

    drift_status = str(
        drift_lookup.get(
            "mean_drift_status",
            "UNKNOWN"
        )
    )

    psi_score = float(
        drift_lookup.get(
            "psi_score",
            0
        )
    )

    psi_status = str(
        drift_lookup.get(
            "psi_status",
            "UNKNOWN"
        )
    )

    # --------------------------------------------------------
    # KPI OVERVIEW
    # --------------------------------------------------------

    st.subheader("Demand Drift Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Reference Avg. Demand",
            f"{reference_average:.2f}"
        )

    with col2:
        st.metric(
            "Current Avg. Demand",
            f"{current_average:.2f}"
        )

    with col3:
        st.metric(
            "Demand Change",
            f"+{demand_change:.2f}%"
        )

    with col4:
        st.metric(
            "PSI Score",
            f"{psi_score:.4f}"
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    st.subheader("Drift Detection Status")

    col1, col2 = st.columns(2)

    with col1:

        if drift_status == "DRIFT_DETECTED":
            st.error(
                "⚠️ Demand-level drift detected"
            )

        else:
            st.success(
                "✅ No major demand-level drift detected"
            )

    with col2:

        if psi_status == "STABLE":
            st.success(
                f"✅ Distribution Status: {psi_status}"
            )

        else:
            st.warning(
                f"⚠️ Distribution Status: {psi_status}"
            )

    # --------------------------------------------------------
    # MODEL RECOMMENDATION
    # --------------------------------------------------------

    st.subheader("Model Recommendation")

    if drift_status == "DRIFT_DETECTED":

        st.warning(
            """
            🔄 **Model retraining is recommended.**

            Average demand has changed materially from the
            reference period. The forecasting pipeline should
            be recalibrated with more recent observations
            before future fleet-planning decisions.
            """
        )

    else:

        st.success(
            """
            Current demand behavior remains sufficiently close
            to the reference period. Immediate retraining is
            not required.
            """
        )

    # --------------------------------------------------------
    # PSI INTERPRETATION
    # --------------------------------------------------------

    st.subheader(
        "Population Stability Index Interpretation"
    )

    st.markdown(
        """
        PSI compares the distribution of demand between a
        reference period and a current period.

        - **PSI < 0.10:** Stable
        - **PSI 0.10–0.25:** Moderate drift
        - **PSI > 0.25:** Significant drift
        """
    )

    st.info(
        f"""
        Current PSI = **{psi_score:.4f}**

        Distribution status: **{psi_status}**

        Although average demand changed by
        **{demand_change:.2f}%**, the PSI result indicates
        that the overall shape of the demand distribution
        remains relatively stable.
        """
    )

    # --------------------------------------------------------
    # SHOCK SCENARIOS
    # --------------------------------------------------------

    st.subheader("Simulated Demand Shock Scenarios")

    col1, col2 = st.columns(2)

    with col1:

        st.warning(
            """
            ❄️ **Snowstorm Scenario**

            Simulated snowstorm conditions reduced demand by
            approximately **34.87%**.

            **Operational response:**

            - reduce unnecessary repositioning,
            - route vehicles toward charging,
            - schedule maintenance,
            - preserve fleet readiness for recovery.
            """
        )

    with col2:

        st.success(
            """
            🎟️ **Major Event Scenario**

            A simulated Manhattan event increased demand by
            approximately **60.01%**.

            **Operational response:**

            - pre-position additional vehicles,
            - protect capacity near event zones,
            - increase short-term fleet availability,
            - monitor demand continuously.
            """
        )

    # --------------------------------------------------------
    # BUSINESS IMPACT
    # --------------------------------------------------------

    st.subheader("Operational Impact")

    st.success(
        f"""
        🚀 Demand monitoring identified a
        **{demand_change:.2f}% increase in average demand**
        between the reference and current periods.

        This monitoring layer can help the fleet operator:

        - trigger model retraining,
        - adjust vehicle positioning,
        - maintain service during demand shocks,
        - move maintenance into low-demand periods,
        - and prepare fleet capacity for major events.
        """
    )