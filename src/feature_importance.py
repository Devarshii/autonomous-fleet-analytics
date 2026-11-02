import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = Path("models/lightgbm_demand_model.pkl")

OUTPUT_DIR = Path("outputs/forecasting")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LIGHTGBM FEATURE IMPORTANCE")
print("=" * 70)

print("\nLoading trained LightGBM model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# GET FEATURE NAMES
# ============================================================

if hasattr(model, "feature_name_"):
    feature_names = model.feature_name_

elif hasattr(model, "feature_name"):
    feature_names = model.feature_name()

else:
    raise ValueError(
        "Could not find feature names in the trained model."
    )


# ============================================================
# GET FEATURE IMPORTANCE
# ============================================================

if hasattr(model, "feature_importances_"):
    importance_values = model.feature_importances_

else:
    raise ValueError(
        "Could not find feature importance values."
    )


importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": importance_values,
    }
)

importance_df = importance_df.sort_values(
    "importance",
    ascending=False,
).reset_index(drop=True)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\nFeature importance ranking:")

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# SELECT TOP FEATURES
# ============================================================

top_features = importance_df.head(12).copy()

top_features = top_features.sort_values(
    "importance",
    ascending=True,
)


# ============================================================
# CREATE CHART
# ============================================================

plt.figure(figsize=(10, 7))

bars = plt.barh(
    top_features["feature"],
    top_features["importance"],
)

plt.title(
    "LightGBM Demand Forecasting\n"
    "Top Feature Importance"
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.grid(
    axis="x",
    alpha=0.3,
)

for bar, value in zip(
    bars,
    top_features["importance"],
):
    plt.text(
        value,
        bar.get_y() + bar.get_height() / 2,
        f" {value:,.0f}",
        va="center",
        fontsize=10,
    )

plt.tight_layout()


# ============================================================
# SAVE CHART
# ============================================================

output_file = (
    OUTPUT_DIR / "lightgbm_feature_importance.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# SAVE FEATURE IMPORTANCE TABLE
# ============================================================

csv_file = (
    OUTPUT_DIR / "lightgbm_feature_importance.csv"
)

importance_df.to_csv(
    csv_file,
    index=False,
)


# ============================================================
# COMPLETE
# ============================================================

print("\nChart saved successfully!")
print(f"Chart: {output_file}")

print("\nFeature importance table saved!")
print(f"Table: {csv_file}")

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("=" * 70)