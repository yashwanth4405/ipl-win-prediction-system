import pandas as pd

# Load datasets
matches = pd.read_csv("data/matches.csv")
deliveries = pd.read_csv("data/deliveries.csv")

# First 5 rows
print("\nMATCHES DATASET")
print(matches.head())

print("\nDELIVERIES DATASET")
print(deliveries.head())

# Column names
print("\nMATCHES COLUMNS")
print(matches.columns)

print("\nDELIVERIES COLUMNS")
print(deliveries.columns)

# Dataset shape
print("\nMATCHES SHAPE")
print(matches.shape)

print("\nDELIVERIES SHAPE")
print(deliveries.shape)