# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:21:25 2026

@author: masci
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and clean
df = pd.read_csv("data/california_housing.csv")
df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace=True)
df.rename(columns={"median_house_value": "price"}, inplace=True)
df = pd.get_dummies(df, columns=["ocean_proximity"], dtype=int)

# Feature engineering
df["rooms_per_household"]       = df["total_rooms"]    / df["households"]
df["bedrooms_per_room"]         = df["total_bedrooms"]  / df["total_rooms"]
df["population_per_household"]  = df["population"]     / df["households"]

print(df.columns.tolist())
print(df.shape)

# Split
TARGET   = "price"
FEATURES = [c for c in df.columns if c != TARGET]

X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape[0]:,} rows")
print(f"Test:  {X_test.shape[0]:,} rows")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Feature engineering complete.")