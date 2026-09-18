import pandas as pd

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_excel("data/ppc_campaign_performance_data.xlsx")

print("Original dataset shape:", df.shape)


# ==========================================
# 2. CONVERT DATE
# ==========================================

df["Date"] = pd.to_datetime(df["Date"])

# Extract month from Date
df["Month"] = df["Date"].dt.month


# ==========================================
# 3. SELECT OUR 8 FEATURES
# ==========================================

features = [
    "Budget",
    "Duration",
    "Platform",
    "Content_Type",
    "Target_Age",
    "Target_Gender",
    "Region",
    "Month"
]

target = "Conversion_Rate"

X = df[features]
y = df[target]


# ==========================================
# 4. DISPLAY THE RESULT
# ==========================================

print("\nSelected Features:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nData types:")
print(X.dtypes)

print("\nMissing values:")
print(X.isnull().sum())
print("\nTarget missing values:", y.isnull().sum())