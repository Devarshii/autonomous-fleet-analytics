"""Integrated Fleet Decision Lab."""

def render():

    from pathlib import Path

    import pandas as pd
    import streamlit as st

    # ============================================================
    # PAGE CONFIGURATION
    # ============================================================


    ROOT = Path(__file__).resolve().parents[1]
    DATA_FILE = (
        ROOT
        / "outputs"
        / "simulation"
        / "geographic_dispatch_summary.csv"
    )

    # ============================================================
    # HEADER
    # ============================================================

    st.title("🚕 Fleet Decision Lab")
    st.subheader("From Demand Forecasting to Business Decisions")

    st.caption(
        "NYC Pilot | January 25–31, 2025 | "
        "1,000 Simulated Vehicles"
    )

    st.info(
        "Explore how fleet positioning strategies affect "
        "simulated passenger service, estimated empty driving, "
        "and modeled business contribution. "
        "Financial assumptions can be adjusted interactively."
    )

    # ============================================================
    # LOAD SIMULATION DATA
    # ============================================================

    if not DATA_FILE.exists():
        st.error(
            "Simulation results not found. Run "
            "`python src/geographic_dispatch_simulator.py` first."
        )
        st.stop()


    @st.cache_data
    def load_data(path, modified_time):
        return pd.read_csv(path)


    df = load_data(
        str(DATA_FILE),
        DATA_FILE.stat().st_mtime,
    )

    required_columns = {
        "policy",
        "requests",
        "served",
        "service_rate_pct",
        "mean_wait_min_served",
        "estimated_total_empty_miles",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        st.error(
            f"Missing simulation columns: "
            f"{sorted(missing_columns)}"
        )
        st.stop()

    if df["policy"].duplicated().any():
        st.error("Duplicate policy rows found in the summary.")
        st.stop()

    df = df.set_index("policy")

    policy_names = {
        "no_repositioning": "No Repositioning",
        "forecast_repositioning": "Forecast-Driven",
        "arrival_aware_repositioning": "Arrival-Aware",
    }

    missing_policies = set(policy_names) - set(df.index)

    if missing_policies:
        st.error(
            f"Missing simulation policies: "
            f"{sorted(missing_policies)}"
        )
        st.stop()

    # ============================================================
    # BUSINESS ASSUMPTIONS
    # ============================================================

    st.divider()

    st.header("1. Business Assumptions")

    col1, col2 = st.columns(2)

    with col1:
        ride_contribution = st.slider(
            "Contribution per completed ride ($)",
            min_value=0.0,
            max_value=20.0,
            value=3.0,
            step=0.25,
            help=(
                "Illustrative contribution per completed ride "
                "before modeled empty-driving costs."
            ),
        )

    with col2:
        mile_cost = st.slider(
            "Operating cost per empty mile ($)",
            min_value=0.0,
            max_value=3.0,
            value=0.50,
            step=0.05,
            help=(
                "Illustrative incremental operating cost "
                "for each estimated empty mile."
            ),
        )

    # ============================================================
    # CALCULATE FINANCIAL PERFORMANCE
    # ============================================================

    results = []

    for policy, display_name in policy_names.items():
        row = df.loc[policy]

        rides = int(row["served"])
        requests = int(row["requests"])
        empty_miles = float(
            row["estimated_total_empty_miles"]
        )

        modeled_contribution = (
            rides * ride_contribution
            - empty_miles * mile_cost
        )

        results.append({
            "Policy": display_name,
            "Requests": requests,
            "Rides Served": rides,
            "Service Rate (%)": float(
                row["service_rate_pct"]
            ),
            "Average Wait (min)": float(
                row["mean_wait_min_served"]
            ),
            "Empty Miles": empty_miles,
            "Modeled Net Contribution ($)": modeled_contribution,
        })

    result_df = pd.DataFrame(results)

    # ============================================================
    # BASELINE COMPARISON
    # ============================================================

    baseline = result_df.loc[
        result_df["Policy"] == "No Repositioning"
    ].iloc[0]

    forecast = result_df.loc[
        result_df["Policy"] == "Forecast-Driven"
    ].iloc[0]

    extra_rides = int(
        forecast["Rides Served"]
        - baseline["Rides Served"]
    )

    extra_miles = float(
        forecast["Empty Miles"]
        - baseline["Empty Miles"]
    )

    incremental_value = (
        extra_rides * ride_contribution
        - extra_miles * mile_cost
    )

    # ============================================================
    # KPIS
    # ============================================================

    st.divider()

    st.header("2. Operational & Financial Performance")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Additional Simulated Rides",
        f"{extra_rides:+,}",
    )

    k2.metric(
        "Additional Estimated Empty Miles",
        f"{extra_miles:+,.0f}",
    )

    k3.metric(
        "Incremental Modeled Contribution",
        f"${incremental_value:+,.0f}",
    )

    k4.metric(
        "Forecast-Driven Service Rate",
        f"{forecast['Service Rate (%)']:.2f}%",
        delta=(
            f"{forecast['Service Rate (%)'] - baseline['Service Rate (%)']:+.2f} pp"
        ),
    )

    # ============================================================
    # POLICY COMPARISON TABLE
    # ============================================================

    st.subheader("Fleet Strategy Comparison")

    st.dataframe(
        result_df.style.format({
            "Requests": "{:,.0f}",
            "Rides Served": "{:,.0f}",
            "Service Rate (%)": "{:.2f}%",
            "Average Wait (min)": "{:.2f}",
            "Empty Miles": "{:,.0f}",
            "Modeled Net Contribution ($)": "${:,.0f}",
        }),
        width="stretch",
        hide_index=True,
    )

    # ============================================================
    # VISUAL COMPARISON
    # ============================================================

    left, right = st.columns(2)

    with left:
        st.subheader("Requests Served by Policy")

        chart_data = result_df.set_index("Policy")[
            ["Rides Served"]
        ]

        st.bar_chart(chart_data)

    with right:
        st.subheader("Estimated Empty Miles by Policy")

        chart_data = result_df.set_index("Policy")[
            ["Empty Miles"]
        ]

        st.bar_chart(chart_data)

    st.subheader("Modeled Net Contribution by Policy")

    contribution_chart = result_df.set_index("Policy")[
        ["Modeled Net Contribution ($)"]
    ]

    st.bar_chart(contribution_chart)

    # ============================================================
    # BREAK-EVEN ANALYSIS
    # ============================================================

    st.divider()

    st.header("3. Break-Even Analysis")

    if extra_rides > 0:
        break_even = (
            extra_miles * mile_cost / extra_rides
        )

        b1, b2 = st.columns(2)

        b1.metric(
            "Break-Even Contribution per Extra Ride",
            f"${break_even:.2f}",
        )

        b2.metric(
            "Selected Contribution per Ride",
            f"${ride_contribution:.2f}",
        )

        if incremental_value > 0:
            st.success(
                "Under these assumptions, the additional "
                "modeled ride contribution exceeds the "
                "additional estimated empty-driving cost."
            )

        elif incremental_value < 0:
            st.warning(
                "Under these assumptions, the additional "
                "estimated empty-driving cost exceeds "
                "the additional modeled ride contribution."
            )

        else:
            st.info(
                "The two strategies are approximately "
                "at the modeled break-even point."
            )

    else:
        st.warning(
            "The forecast-driven strategy does not serve "
            "additional rides under this simulation."
        )

    # ============================================================
    # POLICY RECOMMENDATION
    # ============================================================

    st.divider()

    st.header("4. Business Decision Recommendation")

    best_row = result_df.loc[
        result_df["Modeled Net Contribution ($)"].idxmax()
    ]

    best_policy = best_row["Policy"]
    best_contribution = best_row[
        "Modeled Net Contribution ($)"
    ]

    st.success(
        f"Highest modeled contribution: **{best_policy}**"
    )

    st.metric(
        "Highest Modeled Net Contribution",
        f"${best_contribution:,.0f}",
    )

    st.write(
        "The recommendation is based on the three "
        "simulated strategies and your selected financial "
        "assumptions. Changing the ride contribution or "
        "cost per empty mile may change the recommended policy."
    )

    st.subheader("Business Interpretation")

    if best_policy == "Forecast-Driven":
        st.write(
            "Under these assumptions, the additional "
            "simulated rides served outweigh the increased "
            "estimated empty-driving costs."
        )

    elif best_policy == "Arrival-Aware":
        st.write(
            "Under these assumptions, the arrival-aware "
            "policy provides the highest modeled contribution "
            "among the three strategies."
        )

    else:
        st.write(
            "Under these assumptions, the cost of repositioning "
            "is not justified by the additional modeled rides. "
            "Keeping vehicles in place provides the highest "
            "modeled contribution."
        )

    # ============================================================
    # LIMITATIONS
    # ============================================================

    st.divider()

    st.header("5. Model Assumptions & Limitations")

    st.caption(
        "This Decision Lab uses development-stage simulation "
        "results from NYC taxi trip records for January 25–31, 2025. "
        "Recorded pickup times are treated as request times. "
        "Pickup and repositioning distances are geographic estimates, "
        "not actual routed road mileage."
    )

    st.caption(
        "The simulation assumes a 12-minute maximum pickup wait "
        "and dispatches currently idle vehicles. It does not include "
        "charging, vehicle maintenance, rider cancellations, "
        "actual robotaxi telemetry, or independently validated "
        "operational outcomes."
    )

    st.caption(
        "Financial inputs are hypothetical scenario assumptions. "
        "Modeled net contribution is not observed revenue, "
        "profit, or a complete operating-cost forecast. "
        "Results should not be interpreted as proof of "
        "real-world robotaxi performance."
    )

    st.divider()

    st.caption(
        "Autonomous Fleet Analytics | NYC Pilot | "
        "Interactive Fleet Decision Intelligence"
    )


    st.divider()
    st.header("6. January vs February Validation")

    jan_path = (
        ROOT / "outputs" / "simulation"
        / "geographic_dispatch_summary.csv"
    )
    feb_path = (
        ROOT / "outputs" / "validation"
        / "february_fleet"
        / "february_fleet_summary.csv"
    )

    if jan_path.exists() and feb_path.exists():
        records = []

        for label, path in [
            ("January 25-31", jan_path),
            ("February 1-7", feb_path),
        ]:
            period = pd.read_csv(path).set_index("policy")
            baseline = period.loc["no_repositioning"]
            forecast = period.loc["forecast_repositioning"]

            records.append({
                "Evaluation Week": label,
                "Baseline Service Rate (%)": float(
                    baseline["service_rate_pct"]
                ),
                "Forecast-Driven Service Rate (%)": float(
                    forecast["service_rate_pct"]
                ),
                "Additional Simulated Rides": int(
                    forecast["served"] - baseline["served"]
                ),
                "Additional Estimated Empty Miles": float(
                    forecast["estimated_total_empty_miles"]
                    - baseline["estimated_total_empty_miles"]
                ),
            })

        comparison = pd.DataFrame(records)

        st.dataframe(
            comparison.style.format({
                "Baseline Service Rate (%)": "{:.2f}%",
                "Forecast-Driven Service Rate (%)": "{:.2f}%",
                "Additional Simulated Rides": "{:+,}",
                "Additional Estimated Empty Miles": "{:+,.0f}",
            }),
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "February out-of-time evaluation: the "
            "January-trained LightGBM model did not "
            "outperform the historical baseline on MAE "
            "(3.868 vs. 3.860). Service improvements are "
            "simulated, not measured in real AV operations."
        )
    else:
        st.warning(
            "January-February validation files are missing."
        )
