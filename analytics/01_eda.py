# 01_eda.py
# Titanic EDA, profiling, cleaning, and data story
# Run this file from the /analytics directory.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD DATA ONCE AND SAVE OFFLINE FALLBACK
# ============================================================

print("=" * 70)
print("1. LOAD TITANIC DATASET")
print("=" * 70)

df = sns.load_dataset("titanic")

# Required offline fallback
df.to_csv("titanic.csv", index=False)

print("Dataset loaded from Seaborn.")
print("Offline fallback saved as titanic.csv")
print("Shape:", df.shape)


# ============================================================
# 2. PROFILE DATASET
# ============================================================

print("\n" + "=" * 70)
print("2. DATA PROFILE")
print("=" * 70)

print("\n--- df.info() ---")
df.info()

print("\n--- df.describe() ---")
print(df.describe())

print("\n--- df.shape ---")
print(df.shape)


# ============================================================
# 3. MISSING-VALUE PERCENTAGES
# ============================================================

print("\n" + "=" * 70)
print("3. MISSING VALUES BEFORE CLEANING")
print("=" * 70)

missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
missing_pct_affected = missing_pct[missing_pct > 0]

if missing_pct_affected.empty:
    print("No missing values found.")
else:
    for column, pct in missing_pct_affected.items():
        print(f"{column}: {pct:.2f}%")


# ============================================================
# 4. MISSING-VALUE HANDLING
# Threshold:
# <5%       -> drop rows
# 5%-30%    -> impute
# >30%      -> drop column OR encode missing as category
# ============================================================

print("\n" + "=" * 70)
print("4. CLEANING DECISIONS")
print("=" * 70)

cleaning_decisions = {}

for column, pct in missing_pct_affected.items():

    if pct < 5:
        cleaning_decisions[column] = (
            f"{pct:.2f}% missing (<5%): drop rows containing missing values."
        )

    elif pct <= 30:
        cleaning_decisions[column] = (
            f"{pct:.2f}% missing (5%-30%): impute missing values."
        )

    else:
        cleaning_decisions[column] = (
            f"{pct:.2f}% missing (>30%): encode missing as its own category."
        )

for column, decision in cleaning_decisions.items():
    print(f"{column}: {decision}")

# Work on a cleaned copy while retaining the originally loaded DataFrame.
cleaned_df = df.copy()

# age: 5%-30% -> median imputation
if "age" in cleaned_df.columns and cleaned_df["age"].isna().any():
    cleaned_df["age"] = cleaned_df["age"].fillna(
        cleaned_df["age"].median()
    )

# embarked: <5% -> drop affected rows
if "embarked" in cleaned_df.columns:
    cleaned_df = cleaned_df.dropna(subset=["embarked"])

# embark_town: <5% -> drop affected rows
if "embark_town" in cleaned_df.columns:
    cleaned_df = cleaned_df.dropna(subset=["embark_town"])

# deck: >30% -> preserve missingness as a category
if "deck" in cleaned_df.columns:
    cleaned_df["deck"] = cleaned_df["deck"].astype("object").fillna("Missing")

print("\nMissing values after cleaning:")
print(cleaned_df.isna().sum())

print("\nCleaned shape:", cleaned_df.shape)

# Save cleaned data for the modeling stage.
cleaned_df.to_csv("titanic.csv", index=False)
print("\nCleaned titanic.csv updated.")


# ============================================================
# 5. UNIVARIATE ANALYSIS - AGE
# ============================================================

print("\n" + "=" * 70)
print("5. AGE ANALYSIS")
print("=" * 70)

plt.figure(figsize=(8, 5))
sns.histplot(cleaned_df["age"], bins=30, kde=True)
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("age_histogram.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=cleaned_df["age"])
plt.title("Box Plot of Age")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig("age_boxplot.png", dpi=150)
plt.show()


# ============================================================
# 6. UNIVARIATE ANALYSIS - FARE
# ============================================================

print("\n" + "=" * 70)
print("6. FARE ANALYSIS")
print("=" * 70)

plt.figure(figsize=(8, 5))
sns.histplot(cleaned_df["fare"], bins=30, kde=True)
plt.title("Distribution of Fare")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("fare_histogram.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=cleaned_df["fare"])
plt.title("Box Plot of Fare")
plt.xlabel("Fare")
plt.tight_layout()
plt.savefig("fare_boxplot.png", dpi=150)
plt.show()


# ============================================================
# 7. IQR OUTLIERS
# ============================================================

print("\n" + "=" * 70)
print("7. IQR OUTLIER COUNTS")
print("=" * 70)


def iqr_outlier_details(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (data[column] < lower) | (data[column] > upper)

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower": lower,
        "upper": upper,
        "count": int(mask.sum())
    }


age_outlier = iqr_outlier_details(cleaned_df, "age")
fare_outlier = iqr_outlier_details(cleaned_df, "fare")

print("AGE")
print(f"Q1: {age_outlier['q1']:.4f}")
print(f"Q3: {age_outlier['q3']:.4f}")
print(f"IQR: {age_outlier['iqr']:.4f}")
print(f"Lower bound: {age_outlier['lower']:.4f}")
print(f"Upper bound: {age_outlier['upper']:.4f}")
print(f"Outlier count: {age_outlier['count']}")

print("\nFARE")
print(f"Q1: {fare_outlier['q1']:.4f}")
print(f"Q3: {fare_outlier['q3']:.4f}")
print(f"IQR: {fare_outlier['iqr']:.4f}")
print(f"Lower bound: {fare_outlier['lower']:.4f}")
print(f"Upper bound: {fare_outlier['upper']:.4f}")
print(f"Outlier count: {fare_outlier['count']}")


# ============================================================
# 8. FARE MEAN, MEDIAN, MODE AND SKEWNESS
# ============================================================

print("\n" + "=" * 70)
print("8. FARE SUMMARY")
print("=" * 70)

fare_mean = cleaned_df["fare"].mean()
fare_median = cleaned_df["fare"].median()
fare_mode = cleaned_df["fare"].mode().iloc[0]

print(f"Mean:   {fare_mean:.4f}")
print(f"Median: {fare_median:.4f}")
print(f"Mode:   {fare_mode:.4f}")

if fare_mean > fare_median > fare_mode:
    fare_skew_text = (
        "Fare is right-skewed because mean > median > mode."
    )
elif fare_mean < fare_median < fare_mode:
    fare_skew_text = (
        "Fare is left-skewed because mean < median < mode."
    )
else:
    fare_skew_text = (
        "Fare does not follow a simple mean/median/mode ordering; "
        "inspect the histogram and skewness value."
    )

print(fare_skew_text)
print(f"pandas skewness: {cleaned_df['fare'].skew():.4f}")


# ============================================================
# 9. BIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("9. SURVIVAL RATES")
print("=" * 70)

survival_by_sex = (
    cleaned_df.groupby("sex")["survived"]
    .mean()
    .mul(100)
    .rename("survival_rate_percent")
    .reset_index()
)

print("\n--- Survival by sex ---")
print(survival_by_sex.to_string(index=False))

survival_by_pclass = (
    cleaned_df.groupby("pclass")["survived"]
    .mean()
    .mul(100)
    .rename("survival_rate_percent")
    .reset_index()
)

print("\n--- Survival by pclass ---")
print(survival_by_pclass.to_string(index=False))

survival_by_sex_pclass = (
    cleaned_df.groupby(["sex", "pclass"])["survived"]
    .mean()
    .mul(100)
    .rename("survival_rate_percent")
    .reset_index()
)

print("\n--- Survival by sex and pclass ---")
print(survival_by_sex_pclass.to_string(index=False))


# Required boolean masking examples
female_first_class = cleaned_df[
    (cleaned_df["sex"] == "female") &
    (cleaned_df["pclass"] == 1)
]

female_or_first_class = cleaned_df[
    (cleaned_df["sex"] == "female") |
    (cleaned_df["pclass"] == 1)
]

print("\nBoolean masking example:")
print(
    "Female + first class survival rate: "
    f"{female_first_class['survived'].mean() * 100:.2f}%"
)
print(
    "Female OR first class survival rate: "
    f"{female_or_first_class['survived'].mean() * 100:.2f}%"
)


# ============================================================
# 10. EXACT SIX-COLUMN CORRELATION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("10. CORRELATION MATRIX")
print("=" * 70)

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr_matrix = cleaned_df[correlation_columns].corr()

print(corr_matrix)

plt.figure(figsize=(10, 7))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Titanic Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()


# ============================================================
# 11. TWO STRONGEST CORRELATIONS
# ============================================================

corr_pairs = []

for i in range(len(correlation_columns)):
    for j in range(i + 1, len(correlation_columns)):
        feature_1 = correlation_columns[i]
        feature_2 = correlation_columns[j]
        coefficient = corr_matrix.loc[feature_1, feature_2]

        corr_pairs.append(
            (feature_1, feature_2, coefficient, abs(coefficient))
        )

corr_pairs.sort(key=lambda x: x[3], reverse=True)

print("\nTwo strongest absolute off-diagonal correlations:")
for feature_1, feature_2, coefficient, absolute_value in corr_pairs[:2]:
    print(
        f"{feature_1} vs {feature_2}: "
        f"correlation={coefficient:.4f}, "
        f"absolute={absolute_value:.4f}"
    )


# ============================================================
# 12. MULTIVARIATE DATA STORY - CHART 1
# ============================================================

plt.figure(figsize=(8, 5))
sns.barplot(
    data=cleaned_df,
    x="sex",
    y="survived"
)
plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("story_01_survival_by_sex.png", dpi=150)
plt.show()

print("""
Chart 1 interpretation:
Female passengers have a substantially higher survival rate than male
passengers. This indicates that sex was an important factor associated
with survival outcomes.
""")


# ============================================================
# 13. MULTIVARIATE DATA STORY - CHART 2
# ============================================================

plt.figure(figsize=(8, 5))
sns.barplot(
    data=cleaned_df,
    x="pclass",
    y="survived"
)
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("story_02_survival_by_class.png", dpi=150)
plt.show()

print("""
Chart 2 interpretation:
Survival rates differ across passenger classes, with higher-class
passengers generally showing better survival outcomes. This suggests
that passenger class was associated with access to survival opportunities.
""")


# ============================================================
# 14. MULTIVARIATE DATA STORY - CHART 3
# ============================================================

plt.figure(figsize=(9, 6))
sns.barplot(
    data=cleaned_df,
    x="pclass",
    y="survived",
    hue="sex"
)
plt.title("Survival Rate by Sex and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("story_03_survival_by_sex_class.png", dpi=150)
plt.show()

print("""
Chart 3 interpretation:
Combining sex and passenger class reveals stronger differences than
either variable alone. Female passengers generally have higher survival
rates, while passenger class further differentiates survival outcomes.
""")


# ============================================================
# 15. MULTIVARIATE DATA STORY - CHART 4
# ============================================================

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=cleaned_df,
    x="age",
    y="fare",
    hue="survived"
)
plt.title("Age vs Fare by Survival")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("story_04_age_fare_survival.png", dpi=150)
plt.show()

print("""
Chart 4 interpretation:
The scatter plot combines age, fare and survival. Higher fares are
associated with passenger classes that generally had better survival
outcomes, showing that survival was related to multiple characteristics.
""")


# ============================================================
# 16. EXPLORATORY STANDARDIZATION CHECK
# ============================================================

print("\n" + "=" * 70)
print("16. EXPLORATORY STANDARDIZATION")
print("=" * 70)

eda_scaler = StandardScaler()

standardized = cleaned_df[["age", "fare"]].copy()
standardized[["age", "fare"]] = eda_scaler.fit_transform(
    standardized[["age", "fare"]]
)

print("Before standardization:")
print(cleaned_df[["age", "fare"]].agg(["mean", "std"]))

print("\nAfter standardization:")
print(standardized[["age", "fare"]].agg(["mean", "std"]))

print("""
Interpretation:
The standardized age and fare columns have approximately mean 0 and
standard deviation 1. This is an EDA-stage sanity check only and is
not used as the modeling pipeline's preprocessing.
""")


# ============================================================
# 17. FINAL EDA SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)
print("Created/updated: titanic.csv")
print("Created charts: age/fare plots, correlation heatmap, and 4 data-story charts.")
print("Next step: run 02_modeling.py")
