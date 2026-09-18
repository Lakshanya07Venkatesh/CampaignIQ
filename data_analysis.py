import pandas as pd

# Load the Excel dataset
df = pd.read_excel("data/ppc_campaign_performance_data.xlsx")

# Display the first 5 rows
print("\n===== FIRST 5 ROWS =====")
print(df.head())

# Display dataset size
print("\n===== DATASET SHAPE =====")
print(df.shape)

# Display all column names
print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

# Display information about each column
print("\n===== DATASET INFORMATION =====")
df.info()

# Check for missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())