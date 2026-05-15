# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:22:51 2026

@author: masci
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

# Train best model
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)
preds = model.predict(X_test_scaled)

print(f"MAE: {mean_absolute_error(y_test, preds):.0f}")
print(f"R²:  {r2_score(y_test, preds):.4f}")

# Predictions vs actual
plt.figure(figsize=(7, 7))
plt.scatter(y_test, preds, alpha=0.2, s=8, color="#4A90D9")
plt.plot([0, 500000], [0, 500000], "r--", linewidth=1.5, label="Perfect prediction")
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title(f"Predictions vs actual  |  R²={r2_score(y_test, preds):.3f}", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.show()

