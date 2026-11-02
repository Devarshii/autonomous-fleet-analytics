# 🚕 Autonomous Fleet Analytics

## Demand-Aware Fleet Positioning and Health Analytics for Autonomous Vehicles

This project presents an end-to-end analytics system designed to support autonomous fleet operations using demand forecasting, vehicle positioning, fleet simulation, maintenance analytics, stuck-risk monitoring, and demand-drift detection.

The system uses historical NYC taxi demand as a proxy for autonomous ride demand and demonstrates how analytics can support decisions such as where idle vehicles should wait, when vehicles should be taken offline for maintenance, and how fleet operations should respond to changing demand conditions.

---

# 📌 Project Overview

Autonomous mobility operators must continuously make decisions about:

- where vehicles should be positioned,
- how many vehicles are needed in each area,
- when vehicles can safely be removed for maintenance,
- where operational risk is highest,
- and how unexpected changes in demand should affect fleet planning.

A poorly positioned fleet can result in:

- longer passenger wait times,
- unnecessary empty vehicle travel,
- lower fleet utilization,
- missed ride opportunities,
- increased energy consumption,
- and inefficient maintenance scheduling.

This project builds an analytics pipeline that combines machine learning, optimization, simulation, and operational monitoring to address these problems.

---

# 🎯 Business Problem

An autonomous ride-hailing operator has a limited number of vehicles.

Ride demand changes depending on:

- location,
- hour of the day,
- day of the week,
- weather conditions,
- major events,
- holidays,
- and changing travel behavior.

At the same time, some vehicles may be unavailable because of:

- preventive maintenance,
- charging,
- elevated vehicle-health risk,
- or operational incidents.

The goal is to determine how the fleet can be managed more intelligently while maintaining service availability.

---

# 🎯 Project Objectives

The project focuses on five major objectives:

1. **Forecast ride demand**
   - Predict hourly demand across NYC taxi zones.

2. **Optimize idle vehicle positioning**
   - Allocate available vehicles toward areas with higher expected demand.

3. **Evaluate positioning strategies**
   - Compare a baseline strategy with forecast-driven repositioning through simulation.

4. **Support maintenance decisions**
   - Identify high-risk vehicles and schedule maintenance during lower-demand periods.

5. **Monitor operational risk and demand drift**
   - Identify high-risk operating conditions and detect meaningful changes in demand behavior.

---

# 🏗️ Project Architecture

```text
NYC TLC Trip Data
        │
        ▼
Data Cleaning & Validation
        │
        ▼
Taxi Zone + Time Aggregation
        │
        ▼
Feature Engineering
        │
        ▼
Demand Forecasting
Historical Baseline + LightGBM
        │
        ▼
Demand-Aware Fleet Positioning
        │
        ▼
7-Day Fleet Simulation
        │
        ├───────────────┐
        ▼               ▼
Maintenance       Stuck-Risk
Analytics          Monitoring
        │               │
        └───────┬───────┘
                ▼
        Shock & Drift Analysis
                │
                ▼
         Streamlit Dashboard
```

---

# 📊 Data Sources

## NYC Taxi Demand

The primary demand dataset is based on:

**NYC Taxi & Limousine Commission Yellow Taxi Trip Records**

The dataset contains information such as:

- pickup timestamp,
- drop-off timestamp,
- pickup taxi zone,
- drop-off taxi zone,
- trip distance,
- passenger count,
- fare information,
- payment method.

The project currently uses **January 2025 Yellow Taxi Trip Records** for the NYC pilot analysis.

## NYC Taxi Zones

NYC TLC taxi-zone geographic data is used to connect taxi-zone IDs with:

- borough,
- zone name,
- service zone,
- geographic boundaries.

Because modern NYC TLC trip records provide `PULocationID` rather than exact pickup latitude and longitude, the NYC implementation uses official TLC taxi zones as the primary spatial unit.

---

# 🧹 Data Preparation

The preprocessing pipeline performs several validation and cleaning steps.

Examples include:

- filtering records to the intended analysis period,
- validating pickup and drop-off timestamps,
- identifying invalid trip durations,
- checking trip distances,
- validating taxi-zone IDs,
- aggregating trips into hourly demand,
- merging taxi-zone metadata.

The final demand dataset represents:

```text
Taxi Zone × Hour → Ride Demand
```

---

# 🕒 Feature Engineering

The forecasting dataset contains time-based and historical-demand features such as:

- hour,
- day of week,
- day of month,
- weekend indicator,
- lagged demand,
- rolling demand statistics,
- taxi-zone information.

Lag features allow the forecasting model to learn recurring temporal patterns in ride demand.

---

# 🔮 Demand Forecasting

Two forecasting approaches were evaluated.

## Historical Baseline

A historical demand benchmark was used as the baseline forecasting method.

## LightGBM

A LightGBM regression model was trained using:

- temporal features,
- lag features,
- historical demand patterns,
- taxi-zone information.

A time-based evaluation period was used so that future observations were not used to predict the past.

---

# 📈 Forecasting Results

The historical baseline achieved:

**MAE = 4.40 rides per zone-hour**

The LightGBM model achieved:

**MAE = 3.53 rides per zone-hour**

This represents approximately:

## 🚀 19.9% Lower Forecasting Error

The improvement demonstrates that machine-learning-based forecasting can provide more accurate zone-level demand estimates than a simple historical benchmark.

---

# 🚕 Fleet Positioning

The predicted demand is used to determine where idle autonomous vehicles should be positioned.

A simulated fleet of:

## 1,000 Autonomous Vehicles

is distributed across NYC taxi zones according to expected ride demand.

Higher-demand zones receive a larger number of available vehicles.

Example high-priority positioning zones included:

- East Village
- West Village
- Times Square / Theatre District
- Union Square
- Greenwich Village South
- Midtown Center

The positioning model creates both:

- a geographic fleet-positioning map,
- and a ranked vehicle-allocation chart.

---

# 🗺️ Demand-Aware Positioning Strategy

The positioning approach attempts to reduce the mismatch between:

```text
Vehicle Supply
       vs
Expected Ride Demand
```

The goal is to reduce:

- passenger wait time,
- empty vehicle movement,
- missed ride opportunities,
- and inefficient fleet distribution.

---

# 🧪 Fleet Simulation

A multi-hour simulation was created to compare two positioning strategies.

## Policy A — Baseline Strategy

Vehicles follow the baseline positioning approach.

## Policy B — Forecast-Driven Strategy

Vehicle allocation is adjusted using predicted demand.

The simulation evaluates the strategies across:

## 168 Hours

representing a complete 7-day operating period.

---

# 📊 Simulation Results

### Total simulated demand

**776,144 rides**

### Fleet size

**1,000 vehicles**

### Forecast-driven strategy wins

**137 of 168 hours**

---

# 🎯 Allocation Performance

Baseline average allocation error:

**1.12**

Forecast-driven allocation error:

**0.93**

This represents approximately:

## 🚀 16.9% Lower Allocation Error

The forecast-driven positioning strategy more closely matched vehicle supply with expected passenger demand.

---

# 🚖 Ride Service Performance

Baseline rides served:

**148,220**

Forecast-driven rides served:

**149,552**

Additional rides served:

## +1,332 Rides

---

# 📈 Service Rate

Baseline service rate:

**19.10%**

Forecast-driven service rate:

**19.27%**

Improvement:

**+0.17 percentage points**

Although the simulated fleet is intentionally capacity constrained, the forecast-driven strategy consistently improves service performance relative to the baseline.

---

# 🔧 Maintenance Analytics

The project includes simulated vehicle-health information for:

## 1,000 Vehicles

The maintenance model assigns each vehicle a:

- maintenance-risk score,
- maintenance priority,
- operational status.

---

# 🚘 Fleet Health Results

Available vehicles:

**679**

Vehicles requiring maintenance:

**185**

Urgent maintenance vehicles:

**39**

Average maintenance risk:

**46.7 / 100**

Highest observed maintenance risk:

**77.8 / 100**

High-risk vehicles:

**24**

---

# 🛠️ Maintenance Scheduling

Vehicles requiring service are scheduled into maintenance windows.

The scheduling approach attempts to move maintenance activity toward lower-demand periods whenever possible.

Vehicles scheduled:

**224**

Urgent vehicles scheduled:

**39**

This helps reduce the amount of revenue-generating fleet capacity removed during high-demand periods.

---

# ⚠️ Stuck-Risk Monitoring

The project also contains a stuck-risk analytics module designed to identify operating conditions where an autonomous vehicle may face a higher probability of operational difficulty.

The model assigns a:

```text
Stuck Risk Score
```

to taxi-zone and time combinations.

---

# 🚨 Stuck-Risk Results

Average stuck-risk score:

**56.3 / 100**

Maximum stuck-risk score:

**93.2 / 100**

Critical-risk zone-hour records:

**17,417**

Avoid-idle-positioning recommendations:

**17,417**

High-risk conditions frequently appeared in dense Manhattan areas such as Midtown.

---

# 🚦 Operational Recommendation

When stuck-risk exceeds the project threshold, the system may recommend:

```text
AVOID_IDLE_POSITIONING
```

Instead of staging idle vehicles in high-risk zones, fleet operators can reposition vehicles toward nearby lower-risk areas while maintaining access to ride demand.

---

# 🌨️ Shock & Drift Analysis

Ride-demand patterns can change because of:

- weather,
- major events,
- holidays,
- seasonal behavior,
- transportation disruptions,
- or broader changes in passenger behavior.

The project includes a drift-monitoring layer to detect when recent demand differs materially from historical demand.

---

# 📉 Demand Drift Results

Reference-period average demand:

**15.61**

Current-period average demand:

**18.10**

Demand change:

## +15.98%

The system therefore identifies:

```text
DRIFT_DETECTED
```

and recommends model retraining.

---

# 📐 Population Stability Index

Population Stability Index:

**PSI = 0.0156**

Interpretation:

```text
PSI < 0.10 → Stable
```

Therefore, while the average level of demand increased significantly, the overall distribution shape remained relatively stable.

This demonstrates why multiple drift indicators should be monitored rather than relying on a single metric.

---

# ❄️ Simulated Snowstorm Scenario

A simulated snowstorm produced approximately:

## −34.87% Demand

Operational response could include:

- reducing unnecessary vehicle repositioning,
- sending vehicles for charging,
- scheduling maintenance,
- temporarily reducing active fleet capacity.

---

# 🎟️ Simulated Major Event Scenario

A simulated Manhattan event produced approximately:

## +60.01% Demand

Operational response could include:

- pre-positioning vehicles near expected demand hotspots,
- protecting fleet capacity near event zones,
- increasing short-term fleet availability,
- monitoring demand in real time.

---

# 📊 Streamlit Dashboard

The project includes an interactive Streamlit dashboard with the following pages:

### Executive Overview

High-level KPIs and hourly demand patterns.

### Demand Forecasting

Forecast dataset, actual vs predicted results, and model performance.

### Fleet Positioning

Recommended vehicle allocation and NYC positioning map.

### Simulation Results

Comparison between baseline and forecast-driven fleet strategies.

### Maintenance Analytics

Vehicle health, maintenance risk, and maintenance scheduling.

### Stuck-Risk Monitoring

High-risk zones, zone-hour conditions, and operational recommendations.

### Shock & Drift Analysis

Demand drift monitoring, PSI analysis, and simulated demand shocks.

---

# 💼 Business Impact

The project demonstrates how analytics could help an autonomous ride-hailing operator:

- improve demand forecasting,
- position idle vehicles more effectively,
- serve more passengers,
- reduce supply-demand mismatch,
- protect fleet availability,
- prioritize preventive maintenance,
- avoid high-risk operating conditions,
- respond to sudden demand shocks,
- identify model drift,
- and support future fleet-expansion decisions.

---

# 📌 Key Results

| Metric | Result |
|---|---:|
| LightGBM forecast improvement | **19.9% lower MAE** |
| Baseline forecast MAE | **4.40** |
| LightGBM MAE | **3.53** |
| Simulation horizon | **168 hours** |
| Simulated fleet size | **1,000 vehicles** |
| Forecast strategy wins | **137 hours** |
| Allocation error improvement | **16.9%** |
| Additional rides served | **1,332** |
| Maintenance vehicles scheduled | **224** |
| Urgent maintenance vehicles | **39** |
| Maximum stuck-risk score | **93.2 / 100** |
| Demand drift detected | **+15.98%** |
| PSI score | **0.0156** |
| Snowstorm demand shock | **−34.87%** |
| Major-event demand shock | **+60.01%** |

---

# 🛠️ Technologies Used

## Programming

- Python

## Data Processing

- Pandas
- NumPy
- PyArrow
- DuckDB

## Machine Learning

- Scikit-learn
- LightGBM

## Geographic Analysis

- GeoPandas
- Shapely
- H3

## Optimization

- OR-Tools

## Visualization

- Matplotlib
- Plotly

## Dashboard

- Streamlit

## Development

- VS Code
- Git
- GitHub

---

# 📁 Project Structure

```text
autonomous-fleet-analytics/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── models/
│
├── notebooks/
│
├── outputs/
│   ├── eda/
│   ├── forecasting/
│   ├── fleet_positioning/
│   ├── simulation/
│   ├── maintenance/
│   ├── stuck_risk/
│   └── shock_drift/
│
├── src/
│
├── README.md
└── requirements.txt
```

---

# ▶️ How to Run the Project

## 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd autonomous-fleet-analytics
```

## 2. Create a virtual environment

```bash
python3.12 -m venv .venv
```

Activate the environment on macOS/Linux:

```bash
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

The application should open at:

```text
http://localhost:8501
```

---

# ⚠️ Project Limitations

This project is intended as an analytical and portfolio demonstration.

Several limitations should be considered.

### Human ride-hailing demand as a proxy

NYC taxi demand is used as a proxy for potential autonomous ride demand.

Actual robotaxi demand may differ from conventional taxi demand.

### Taxi-zone spatial resolution

Modern NYC TLC trip records provide taxi-zone IDs rather than exact pickup coordinates.

Therefore, the NYC analysis uses official TLC taxi zones as the spatial unit.

### Simulated autonomous vehicle telemetry

Vehicle-health, maintenance, and several operational-risk variables are simulated.

They do not represent production telemetry from an autonomous vehicle operator.

### Simulated fleet operations

The fleet-positioning and service simulation simplifies real-world constraints such as:

- traffic,
- travel time,
- charging availability,
- road closures,
- vehicle routing,
- passenger cancellations,
- and real-time dispatch behavior.

---

# 🔮 Future Improvements

Possible extensions include:

- Chicago rideshare validation,
- multi-city forecasting,
- national transfer-learning model,
- real-time weather integration,
- live event data,
- traffic-aware repositioning,
- EV charging constraints,
- min-cost-flow or mixed-integer optimization,
- real road-network travel times,
- automated model retraining,
- anomaly detection from real telemetry,
- interactive map-based vehicle simulation.

---

# 🌎 Future National Extension

The broader project architecture can be extended beyond NYC using:

- U.S. Census ACS,
- LODES employment data,
- OpenStreetMap,
- NOAA weather,
- transit accessibility,
- points of interest,
- charger availability,
- and city-level mobility datasets.

A city-holdout evaluation strategy could be used to test whether demand patterns learned in one city can generalize to previously unseen markets.

---

# 🧠 Project Takeaway

The project demonstrates that autonomous fleet management is not only a forecasting problem.

Effective fleet operations require several analytical components working together:

```text
Forecast Demand
        ↓
Position Vehicles
        ↓
Evaluate Strategy
        ↓
Monitor Vehicle Health
        ↓
Monitor Operational Risk
        ↓
Detect Demand Changes
        ↓
Continuously Improve Decisions
```

By integrating these components into one system, fleet operators can make more informed decisions about vehicle availability, positioning, maintenance, and operational readiness.

---

# 📌 Disclaimer

NYC taxi demand is based on public transportation data.

Autonomous-vehicle telemetry, maintenance data, and certain operational-risk variables used in this project are simulated for educational and analytical purposes.

The project should not be interpreted as representing the internal operations, systems, or performance of any real autonomous vehicle company.