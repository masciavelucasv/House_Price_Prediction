# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:22:02 2026

@author: masci
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

os.makedirs("outputs", exist_ok=True)

# Load and clean
df = pd.read_csv("data/california_housing.csv")
df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace=True)
df.rename(columns={"median_house_value": "price"}, inplace=True)
df = pd.get_dummies(df, columns=["ocean_proximity"], dtype=int)

# Feature engineering
df["rooms_per_household"]      = df["total_rooms"]   / df["households"]
df["bedrooms_per_room"]        = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"]    / df["households"]

# Split and scale
TARGET   = "price"
FEATURES = [c for c in df.columns if c != TARGET]
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train models
models = {
    "Random forest":     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42),
    "Ridge regression":  Ridge(alpha=10),
}

results = []
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2})
    print(f"  MAE={mae:.0f}  RMSE={rmse:.0f}  R²={r2:.4f}")

results_df = pd.DataFrame(results).sort_values("R²", ascending=False)
print(results_df)

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Model comparison", fontsize=13, fontweight="bold")
colors  = ["#4A90D9", "#E8593C", "#2ECC71"]
metrics = ["MAE", "RMSE", "R²"]
for ax, metric in zip(axes, metrics):
    vals   = results_df[metric].values
    labels = results_df["Model"].values
    ax.bar(labels, vals, color=colors, edgecolor="white")
    ax.set_title(metric, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.01, f"{v:.0f}" if metric != "R²" else f"{v:.3f}",
                ha="center", fontsize=8)
plt.tight_layout()
plt.show()