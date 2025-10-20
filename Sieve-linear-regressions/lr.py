# ===============================
# Predict "Score" from 18 binary variables
# ===============================

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ----------------------------------------
# 1. Create your dataset
# ----------------------------------------

data = {
    "Score": [69.5, -27.5, 109, -38.5, -107, 121.5, 86.5, 79, 96, 113, 73, 114, 65, -108],
    "A": [True, True, True, True, True, True, True, True, True, True, False, False, False, False],
    "B": [True, True, True, True, True, False, False, False, False, False, False, False, False, False],
    "C": [True, True, True, True, False, False, False, False, False, False, True, False, False, False],
    "D": [True, False, False, False, False, False, False, True, True, False, False, False, False, False],
    "E": [True, True, True, False, False, False, False, False, False, False, False, False, False, False],
    "F": [True, False, False, False, True, False, False, False, False, False, False, False, False, False],
    "G": [True, True, False, False, False, False, False, False, False, False, False, False, False, False],
    "H": [True, False, False, True, False, False, False, False, False, False, False, False, False, False],
    "I": [True, False, True, False, False, False, False, False, False, False, False, False, False, False],
    "J": [False, True, False, False, False, False, False, False, False, True, False, False, False, False],
    "K": [False, False, False, True, False, False, True, False, False, False, False, False, False, False],
    "L": [False, False, False, True, False, True, False, False, False, False, False, False, False, False],
    "M": [False, False, True, False, False, True, False, False, False, False, False, False, False, False],
    "N": [False, True, False, False, False, False, False, False, False, False, False, False, False, False],
    "O": [False, False, False, True, False, False, False, False, False, False, False, False, False, False],
    "P": [False, False, False, False, False, False, True, False, False, False, False, False, False, False],
    "Q": [False, False, False, False, False, False, True, False, False, False, False, False, False, False],
    "R": [False, False, False, False, False, True, False, False, False, False, False, False, False, False],
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

plt.figure(figsize=(6,6))
plt.scatter(y, y_pred_rf, label="Predicted (RF)")
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label="Perfect Fit")
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted Scores (Random Forest)")
plt.legend()
plt.tight_layout()
plt.savefig("score_fit_plot.png", dpi=300)  # Save instead of show()