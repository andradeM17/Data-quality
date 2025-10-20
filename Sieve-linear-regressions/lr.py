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

data = {
    "Score": [69.5, -27.5, 109, -38.5, -107, 121.5, 86.5, 79, 96, 113, 73, 114, 65, -108],
    "LANGD": [True, True, True, True, True, True, True, True, True, True, False, False, False, False],
    "DEDUP": [True, True, True, True, True, False, False, False, False, False, False, False, False, False],
    "RMKUP": [True, True, True, True, False, False, False, False, False, False, True, False, False, False],
    "RMLEN": [True, False, False, False, False, False, False, True, True, False, False, False, False, False],
    "RFDOC": [True, True, True, False, False, False, False, False, False, False, False, False, False, False],
    "RMBOL": [True, False, False, False, True, False, False, False, False, False, False, False, False, False],
    "RMOBS": [True, True, False, False, False, False, False, False, False, False, False, False, False, False],
    "METCL": [True, False, False, True, False, False, False, False, False, False, False, False, False, False],
    "RMURL": [True, False, True, False, False, False, False, False, False, False, False, False, False, False],
    "RMCLN": [False, True, False, False, False, False, False, False, False, True, False, False, False, False],
    "SNTSP": [False, False, False, True, False, False, True, False, False, False, False, False, False, False],
    "RMLNG": [False, False, False, True, False, True, False, False, False, False, False, False, False, False],
    "PARSP": [False, False, True, False, False, True, False, False, False, False, False, False, False, False],
    "DOMFX": [False, True, False, False, False, False, False, False, False, False, False, False, False, False],
    "TRANL": [False, False, False, True, False, False, False, False, False, False, False, False, False, False],
    "SPCCR": [False, False, False, False, False, False, True, False, False, False, False, False, False, False],
    "ADDMT": [False, False, False, False, False, False, True, False, False, False, False, False, False, False],
    "WDSEG": [False, False, False, False, False, True, False, False, False, False, False, False, False, False],
}

df = pd.DataFrame(data)

# Convert TRUE/FALSE to 1/0
df = df.replace({True: 1, False: 0})

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

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# 1. Linear Regression Coefficients
# ---------------------------

coefficients = linreg.coef_  # Correct reference to your linear model
features = X.columns

# Sort by coefficient magnitude
indices = np.argsort(coefficients)
sorted_features = [features[i] for i in indices]
sorted_coefficients = coefficients[indices]

plt.figure(figsize=(10, 6))
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
plt.savefig("linear_coefficients.png", dpi=300)
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

plt.figure(figsize=(10, 6))
bars = plt.barh(sorted_features[::-1], sorted_importances[::-1], color='mediumseagreen')
plt.xlabel("Feature Importance")
plt.ylabel("Variable")
plt.title("Random Forest Feature Importances")

# Annotate values
for bar in bars:
    plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():.3f}", va='center')

plt.tight_layout()
plt.savefig("rf_feature_importances.png", dpi=300)
print("Random Forest feature importance plot saved as 'rf_feature_importances.png'")
