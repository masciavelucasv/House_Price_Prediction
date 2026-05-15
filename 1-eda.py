# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:19:45 2026

@author: masci
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load data
df = pd.read_csv("data/california_housing.csv")
df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace=True)
df.rename(columns={"median_house_value": "price"}, inplace=True)

print(df.shape)
print(df.isnull().sum())
print(df.describe())

# Price distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Target variable: House price", fontsize=13, fontweight="bold")
axes[0].hist(df["price"], bins=50, color="#4A90D9", edgecolor="white")
axes[0].set_title("Raw distribution")
axes[0].set_xlabel("Price (USD)")
axes[0].set_ylabel("Count")
axes[1].hist(np.log1p(df["price"]), bins=50, color="#E8593C", edgecolor="white")
axes[1].set_title("Log-transformed")
axes[1].set_xlabel("log(Price + 1)")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
df_encoded = pd.get_dummies(df, columns=["ocean_proximity"], dtype=int)
corr = df_encoded.corr(numeric_only=True).round(2)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Feature correlation matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# Income vs price
plt.figure(figsize=(8, 5))
plt.scatter(df["median_income"], df["price"], alpha=0.2, s=8, color="#4A90D9")
plt.xlabel("Median income")
plt.ylabel("House price (USD)")
plt.title("Income vs Price", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
