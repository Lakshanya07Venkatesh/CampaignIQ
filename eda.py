import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATASET
df = pd.read_excel("data/ppc_campaign_performance_data.xlsx")

# CONVERT DATE
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month

print("===== DATASET SUMMARY =====")
print(df.describe())

print("\n===== CATEGORICAL VALUES =====")

for column in [
    "Platform",
    "Content_Type",
    "Target_Age",
    "Target_Gender",
    "Region"
]:
    print(f"\n{column}:")
    print(df[column].value_counts())

# -------------------------------
# TARGET DISTRIBUTION
# -------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(df["Conversion_Rate"], kde=True)
plt.title("Conversion Rate Distribution")
plt.xlabel("Conversion Rate")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# CORRELATION
# -------------------------------

numeric_columns = [
    "Budget",
    "Clicks",
    "CTR",
    "CPC",
    "Conversions",
    "CPA",
    "Conversion_Rate",
    "Duration",
    "Revenue",
    "Spend",
    "ROAS",
    "Impressions"
]

plt.figure(figsize=(12, 8))
sns.heatmap(
    df[numeric_columns].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Matrix")
plt.show()

# -------------------------------
# PLATFORM
# -------------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Platform",
    y="Conversion_Rate"
)
plt.title("Conversion Rate by Platform")
plt.xticks(rotation=30)
plt.show()

# -------------------------------
# CONTENT TYPE
# -------------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Content_Type",
    y="Conversion_Rate"
)
plt.title("Conversion Rate by Content Type")
plt.show()

# -------------------------------
# AGE GROUP
# -------------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Target_Age",
    y="Conversion_Rate"
)
plt.title("Conversion Rate by Target Age")
plt.xticks(rotation=30)
plt.show()

# -------------------------------
# BUDGET VS CONVERSION RATE
# -------------------------------

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Budget",
    y="Conversion_Rate"
)
plt.title("Budget vs Conversion Rate")
plt.show()

# -------------------------------
# DURATION VS CONVERSION RATE
# -------------------------------

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Duration",
    y="Conversion_Rate"
)
plt.title("Duration vs Conversion Rate")
plt.show()

print("\nEDA COMPLETE")