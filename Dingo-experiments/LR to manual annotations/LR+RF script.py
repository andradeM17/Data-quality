# ===============================
# Predict "Score" from 18 binary variables
# ===============================

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------
# 1. Create your dataset
# ----------------------------------------

df = pd.read_csv("Dingo-experiments/LR to manual annotations/lr-data.csv", header=None)
df = df.set_index(0)
df = df.transpose()
df.columns.name = None
df = df.replace({"TRUE": 1, "FALSE": 0, True: 1, False: 0})
df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

# ----------------------------------------
# 2. Separate features and target
# ----------------------------------------

X = df.drop(columns=["Score"])
y = df["Score"]

# ----------------------------------------
# 3. Linear Regression
# ----------------------------------------

linreg = LinearRegression()
linreg.fit(X, y)

print("=== Linear Regression Coefficients ===")
for name, coef in zip(X.columns, linreg.coef_):
    print(f"{name:6s}: {coef: .3f}")
print("Intercept:", linreg.intercept_)

y_pred_lin = linreg.predict(X)
print(f"R² (Linear): {r2_score(y, y_pred_lin):.3f}\n")

# ----------------------------------------
# 4. Random Forest Regression
# ----------------------------------------

rf = RandomForestRegressor(n_estimators=500, random_state=42)
rf.fit(X, y)
y_pred_rf = rf.predict(X)
print(f"R² (Random Forest): {r2_score(y, y_pred_rf):.3f}")

# Feature importance
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n=== Random Forest Feature Importances ===")
print(importances)

# ----------------------------------------
# 5. Visualization
# ----------------------------------------



# ---------------------------
# 1. Linear Regression Coefficients
# ---------------------------

coefficients = linreg.coef_  # Correct reference to your linear model
features = X.columns

# Sort by coefficient magnitude
indices = np.argsort(coefficients)
sorted_features = [features[i] for i in indices]
sorted_coefficients = coefficients[indices]

plt.figure(figsize=(20, 12))
bars = plt.barh(sorted_features, sorted_coefficients, color=['tomato' if c < 0 else 'skyblue' for c in sorted_coefficients])
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel("Coefficient Value")
plt.ylabel("Variable")
plt.title("Linear Regression Coefficients")

# Annotate values
for bar in bars:
    plt.text(bar.get_width() + (0.5 if bar.get_width() > 0 else -0.5),
             bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():.2f}",
             va='center', ha='right' if bar.get_width() < 0 else 'left')

plt.tight_layout()
plt.savefig("Dingo-experiments/LR to manual annotations/linear_coefficients.png", dpi=300)
print("Linear regression coefficient plot saved as 'linear_coefficients.png'")

# ---------------------------
# 2. Random Forest Feature Importances
# ---------------------------

rf_importances = rf.feature_importances_
features = X.columns

# Sort by importance
indices = np.argsort(rf_importances)[::-1]
sorted_features = [features[i] for i in indices]
sorted_importances = rf_importances[indices]

plt.figure(figsize=(20, 12))
bars = plt.barh(sorted_features[::-1], sorted_importances[::-1], color='mediumseagreen')
plt.xlabel("Feature Importance")
plt.ylabel("Variable")
plt.title("Random Forest Feature Importances")

# Annotate values
for bar in bars:
    plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():.3f}", va='center')

plt.tight_layout()
plt.savefig("Dingo-experiments/LR to manual annotations/rf_feature_importances.png", dpi=300)
print("Random Forest feature importance plot saved as 'rf_feature_importances.png'")

# ----------------------------------------
# 6. Combination Chart: length = LR coef, height = RF importance
# ----------------------------------------

indices = np.argsort(linreg.coef_)
sorted_features = [X.columns[i] for i in indices]
sorted_lr = linreg.coef_[indices]
sorted_rf = rf.feature_importances_[indices]

# 🔹 Remove features where lr == 0 OR rf_val == 0
filtered_data = [
    (feat, lr, rf_val)
    for feat, lr, rf_val in zip(sorted_features, sorted_lr, sorted_rf)
    if lr != 0 and rf_val != 0
]

# Unzip filtered values
filtered_features, filtered_lr, filtered_rf = zip(*filtered_data)

fig, ax = plt.subplots(figsize=(20, 24))
y_positions = np.arange(len(filtered_features))

for y, lr, rf_val in zip(y_positions, filtered_lr, filtered_rf):
    color = 'tomato' if lr < 0 else 'skyblue'
    ax.barh(y, lr, height=rf_val * 5, color=color)

ax.axvline(0, color='black', linewidth=0.8)
ax.set_yticks(y_positions)
ax.tick_params(axis='x', labelsize=50)
ax.set_yticklabels(filtered_features, fontsize=40)
ax.set_xlabel("Linear Regression Coefficient", fontsize=40)
ax.set_title("Feature Importance", fontsize=60)

plt.tight_layout()
plt.savefig("Dingo-experiments/LR to manual annotations/dcomb-double.png", dpi=300)
print("Combination chart saved.")