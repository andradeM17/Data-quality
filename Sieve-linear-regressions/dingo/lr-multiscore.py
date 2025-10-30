import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------------------------
# 1. Load and prepare dataset
# ----------------------------------------
df = pd.read_csv("Sieve-linear-regressions/dingo/lrdata.csv", header=None)
df = df.set_index(0).transpose()
df.columns.name = None
df = df.replace({"TRUE": 1, "FALSE": 0, True: 1, False: 0})
df = df.apply(pd.to_numeric, errors="ignore")

# Identify all "Score" columns
score_cols = [col for col in df.columns if "Rule" in str(col)]

# Ensure output folder exists
output_dir = "Sieve-linear-regressions/dingo"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------------------
# 2. Loop through each Score column
# ----------------------------------------
summary_rows = []

for score_col in score_cols:
    print(f"\n===============================")
    print(f"Running models for target: {score_col}")
    print(f"===============================\n")

    # Split into features and target
    X = df.drop(columns=score_cols)  # use non-score columns as predictors
    y = pd.to_numeric(df[score_col], errors="coerce")

    # Ensure feature names are strings
    X.columns = X.columns.astype(str)

    # --- Linear Regression ---
    linreg = LinearRegression()
    linreg.fit(X, y)
    y_pred_lin = linreg.predict(X)
    r2_lin = r2_score(y, y_pred_lin)

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=500, random_state=42)
    rf.fit(X, y)
    y_pred_rf = rf.predict(X)
    r2_rf = r2_score(y, y_pred_rf)

    # --- Coefficients & Importances ---
    lin_coefs = pd.Series(linreg.coef_, index=X.columns)
    rf_importances = pd.Series(rf.feature_importances_, index=X.columns)

    # ----------------------------------------
    # 3. Visualization
    # ----------------------------------------
    # --- Linear Regression Coefficients ---
    indices = np.argsort(lin_coefs.values)
    sorted_features = lin_coefs.index[indices]
    sorted_coefficients = lin_coefs.values[indices]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_features, sorted_coefficients,
                    color=['tomato' if c < 0 else 'skyblue' for c in sorted_coefficients])
    plt.axvline(0, color='black', linewidth=0.8)
    plt.xlabel("Coefficient Value")
    plt.ylabel("Variable")
    plt.title(f"Linear Regression Coefficients – {score_col}")

    for bar in bars:
        plt.text(bar.get_width() + (0.5 if bar.get_width() > 0 else -0.5),
                 bar.get_y() + bar.get_height()/2,
                 f"{bar.get_width():.2f}",
                 va='center',
                 ha='right' if bar.get_width() < 0 else 'left')

    plt.tight_layout()
    linplot_path = os.path.join(output_dir, f"Charts/{score_col}_linear_coefficients.png")
    plt.savefig(linplot_path, dpi=300)
    plt.close()
    print(f"Saved: {linplot_path}")

    # --- Random Forest Feature Importances ---
    sorted_rf = rf_importances.sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_rf.index[::-1], sorted_rf.values[::-1], color='mediumseagreen')
    plt.xlabel("Feature Importance")
    plt.ylabel("Variable")
    plt.title(f"Random Forest Feature Importances – {score_col}")

    for bar in bars:
        plt.text(bar.get_width() + 0.002,
                 bar.get_y() + bar.get_height()/2,
                 f"{bar.get_width():.3f}",
                 va='center')

    plt.tight_layout()
    rfplot_path = os.path.join(output_dir, f"Charts/{score_col}_rf_feature_importances.png")
    plt.savefig(rfplot_path, dpi=300)
    plt.close()
    print(f"Saved: {rfplot_path}")

    # ----------------------------------------
    # 4. Build summary row
    # ----------------------------------------
    row = {
        "Target": score_col,
        "R2_Linear": r2_lin,
        "R2_RandomForest": r2_rf
    }

    # Add all coefficients and importances
    for feature in X.columns:
        row[f"{feature}_coef"] = lin_coefs.get(feature, np.nan)
        row[f"{feature}_importance"] = rf_importances.get(feature, np.nan)

    summary_rows.append(row)

# ----------------------------------------
# 5. Save wide-format summary CSV
# ----------------------------------------
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(output_dir, "model_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n✅ All models complete.")
print(f"📊 Full summary (coefficients + importances) saved to: {summary_path}")
print(summary_df.head())
