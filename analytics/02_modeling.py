# 02_modeling.py
# Titanic predictive modeling, imbalance handling, tuning,
# regression, model comparison, and pipeline persistence.
#
# This script MUST be run from the same /analytics directory after
# 01_eda.py has produced titanic.csv.
#
# It deliberately does NOT call sns.load_dataset("titanic").
# The raw dataset is loaded from the committed titanic.csv fallback.

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# 1. PATHS
# ============================================================

# Fix: Use Path('.') instead of Path(__file__).resolve().parent for Colab compatibility
BASE_DIR = Path('.')
DATA_PATH = BASE_DIR / "titanic.csv"

print("=" * 70)
print("TITANIC MODELING PIPELINE")
print("=" * 70)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "titanic.csv was not found. Run 01_eda.py first."
    )


# ============================================================
# 2. LOAD THE SAME CLEANED CSV
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded from titanic.csv")
print("Shape:", df.shape)
print(df.head())


# ============================================================
# 3. CLASSIFICATION TARGET AND FEATURES
# ============================================================

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

target = "survived"

X = df[features].copy()
y = df[target].copy()


# ============================================================
# 4. STRATIFIED TRAIN/TEST SPLIT FIRST
# ============================================================

print("\n" + "=" * 70)
print("4. STRATIFIED TRAIN/TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nOverall class balance:")
print(y.value_counts(normalize=True).sort_index())

print("\nTraining class balance:")
print(y_train.value_counts(normalize=True).sort_index())

print("\nTesting class balance:")
print(y_test.value_counts(normalize=True).sort_index())

print("""
Why stratification matters:
The Titanic target contains two classes with unequal proportions.
Stratification keeps approximately the same survived/not-survived
distribution in train and test, making model evaluation more reliable.
""")


# ============================================================
# 5. PREPROCESSING
# Fit ONLY through the training pipeline.
# ============================================================

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocessing:")
print("- Numeric missing values: median imputation")
print("- Categorical missing values: most-frequent imputation")
print("- Categorical encoding: one-hot encoding")
print("- Numeric scaling: StandardScaler")
print("- All fitted preprocessing is contained inside model pipelines.")


# ============================================================
# 6. CLASSIFIERS
# ============================================================

models = {
    "Logistic Regression": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(max_iter=1000)
            )
        ]
    ),

    "Decision Tree": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                DecisionTreeClassifier(
                    random_state=42,
                    max_depth=5
                )
            )
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42
                )
            )
        ]
    )
}


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================

def evaluate_classifier(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "AUC": roc_auc_score(y_test, probabilities),
        "Confusion Matrix": confusion_matrix(y_test, predictions)
    }


# ============================================================
# 8. TRAIN ALL THREE CLASSIFIERS
# ============================================================

print("\n" + "=" * 70)
print("8. TRAIN THREE CLASSIFIERS")
print("=" * 70)

results = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    results[name] = evaluate_classifier(
        model,
        X_test,
        y_test
    )


# ============================================================
# 9. CLASSIFIER COMPARISON TABLE
# ============================================================

classification_table = pd.DataFrame({
    name: {
        "Accuracy": result["Accuracy"],
        "Precision": result["Precision"],
        "Recall": result["Recall"],
        "F1": result["F1"],
        "AUC": result["AUC"]
    }
    for name, result in results.items()
}).T

print("\n" + "=" * 70)
print("9. CLASSIFICATION COMPARISON")
print("=" * 70)
print(classification_table.to_string(float_format=lambda x: f"{x:.4f}"))

classification_table.to_csv(
    BASE_DIR / "classification_results.csv"
)


# ============================================================
# 10. CONFUSION MATRICES
# ============================================================

for name, result in results.items():

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        result["Confusion Matrix"],
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    safe_name = (
        name.lower()
        .replace(" ", "_")
    )

    plt.savefig(
        BASE_DIR / f"confusion_matrix_{safe_name}.png",
        dpi=150
    )

    plt.show()


# ============================================================
# 11. ROC CURVE / AUC
# ============================================================

plt.figure(figsize=(9, 6))

for name, model in models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    auc_value = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc_value:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()

plt.savefig(
    BASE_DIR / "roc_curve_comparison.png",
    dpi=150
)

plt.show()


# ============================================================
# 12. DECISION TREE VISUALIZATION
# ============================================================

tree_pipeline = models["Decision Tree"]

fitted_tree_preprocessor = (
    tree_pipeline.named_steps["preprocessor"]
)

tree_model = tree_pipeline.named_steps["model"]

tree_feature_names = (
    fitted_tree_preprocessor
    .get_feature_names_out()
)

plt.figure(figsize=(25, 12))

plot_tree(
    tree_model,
    feature_names=tree_feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    max_depth=3
)

plt.title("Decision Tree - First Three Levels")
plt.tight_layout()

plt.savefig(
    BASE_DIR / "decision_tree.png",
    dpi=150
)

plt.show()


# ============================================================
# 13. CLASS IMBALANCE
# ============================================================

print("\n" + "=" * 70)
print("13. CLASS IMBALANCE")
print("=" * 70)

class_balance = (
    y.value_counts()
    .sort_index()
    .rename_axis("survived")
    .reset_index(name="count")
)

class_balance["percentage"] = (
    class_balance["count"] / len(y) * 100
)

print(class_balance.to_string(index=False))


# ============================================================
# 14. IMBALANCE STRATEGY COMPARISON
# Baseline vs class_weight='balanced' vs SMOTE
# ============================================================

baseline_rf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

balanced_rf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

smote_rf = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

imbalance_models = {
    "Baseline": baseline_rf,
    "Class Weight Balanced": balanced_rf,
    "SMOTE": smote_rf
}

imbalance_rows = []

for name, model in imbalance_models.items():

    print(f"\nTraining imbalance strategy: {name}")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    imbalance_rows.append({
        "Strategy": name,
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        )
    })

imbalance_table = pd.DataFrame(imbalance_rows)

print("\n" + "=" * 70)
print("IMBALANCE COMPARISON")
print("=" * 70)
print(
    imbalance_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

imbalance_table.to_csv(
    BASE_DIR / "imbalance_results.csv",
    index=False
)

best_imbalance = imbalance_table.loc[
    imbalance_table["F1"].idxmax()
]

print("\nBest strategy by F1:")
print(best_imbalance.to_string())

print("""
Conclusion:
The strategy with the highest F1 provides the best balance between
precision and recall on this test set. SMOTE is applied only within
the training pipeline, so the test set remains untouched.
""")


# ============================================================
# 15. RANDOM FOREST GRID SEARCH
# ============================================================

print("\n" + "=" * 70)
print("15. RANDOM FOREST GRID SEARCH")
print("=" * 70)

rf_for_grid = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                oob_score=True
            )
        )
    ]
)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    estimator=rf_for_grid,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation F1:")
print(f"{grid_search.best_score_:.4f}")

best_rf_pipeline = grid_search.best_estimator_

best_rf_model = (
    best_rf_pipeline.named_steps["model"]
)

print("\nOOB score:")
print(f"{best_rf_model.oob_score_:.4f}")


# ============================================================
# 16. EVALUATE TUNED RANDOM FOREST
# ============================================================

tuned_rf_results = evaluate_classifier(
    best_rf_pipeline,
    X_test,
    y_test
)

print("\nTuned Random Forest test metrics:")
for metric in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUC"
]:
    print(
        f"{metric}: "
        f"{tuned_rf_results[metric]:.4f}"
    )


# ============================================================
# 17. REGRESSION SIDE TASK
# Predict fare from other available features
# ============================================================

print("\n" + "=" * 70)
print("17. FARE REGRESSION")
print("=" * 70)

regression_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked"
]

X_reg = df[regression_features].copy()
y_reg = df["fare"].copy()

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

reg_numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch"
]

reg_categorical_features = [
    "sex",
    "embarked"
]

reg_numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

reg_categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            reg_numeric_transformer,
            reg_numeric_features
        ),
        (
            "cat",
            reg_categorical_transformer,
            reg_categorical_features
        )
    ]
)

regression_pipeline = Pipeline(
    steps=[
        ("preprocessor", reg_preprocessor),
        ("model", LinearRegression())
    ]
)

regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)

y_reg_pred = regression_pipeline.predict(
    X_reg_test
)

mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred
    )
)

r2 = r2_score(
    y_reg_test,
    y_reg_pred
)

X_reg_test_transformed = (
    regression_pipeline
    .named_steps["preprocessor"]
    .transform(X_reg_test)
)

n = len(y_reg_test)
p = X_reg_test_transformed.shape[1]

if n - p - 1 > 0:
    adjusted_r2 = (
        1
        - ((1 - r2) * (n - 1))
        / (n - p - 1)
    )
else:
    adjusted_r2 = np.nan

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Adjusted R²: {adjusted_r2:.4f}")


# ============================================================
# 18. RESIDUAL PLOT
# ============================================================

residuals = y_reg_test - y_reg_pred

plt.figure(figsize=(9, 6))

sns.scatterplot(
    x=y_reg_pred,
    y=residuals
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Fare Regression Residual Plot")
plt.tight_layout()

plt.savefig(
    BASE_DIR / "fare_regression_residuals.png",
    dpi=150
)

plt.show()

print("""
Residual interpretation:
A random residual cloud around zero with approximately constant spread
suggests little visual evidence of heteroscedasticity. A systematic
increase or decrease in residual spread as predicted fare changes would
indicate heteroscedasticity. Use the displayed residual plot together
with this criterion when writing the final conclusion.
""")


# ============================================================
# 19. FINAL MODEL COMPARISON
# Classification and regression are separate metric groups.
# ============================================================

print("\n" + "=" * 70)
print("19. FINAL MODEL COMPARISON")
print("=" * 70)

final_classification = classification_table.copy()

# Include tuned RF as an additional useful reference.
final_classification.loc[
    "Tuned Random Forest",
    ["Accuracy", "Precision", "Recall", "F1", "AUC"]
] = [
    tuned_rf_results["Accuracy"],
    tuned_rf_results["Precision"],
    tuned_rf_results["Recall"],
    tuned_rf_results["F1"],
    tuned_rf_results["AUC"]
]

print("\nCLASSIFICATION METRICS")
print(
    final_classification.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)

regression_metrics = pd.DataFrame({
    "Regression Metric": [
        "MAE",
        "RMSE",
        "R²",
        "Adjusted R²"
    ],
    "Linear Regression": [
        mae,
        rmse,
        r2,
        adjusted_r2
    ]
})

print("\nREGRESSION METRICS")
print(
    regression_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

final_classification.to_csv(
    BASE_DIR / "final_classification_comparison.csv"
)

regression_metrics.to_csv(
    BASE_DIR / "regression_results.csv",
    index=False
)


# ============================================================
# 20. FINAL DEPLOYMENT RECOMMENDATION
# Use F1 as the primary selection criterion, then AUC/recall.
# ============================================================

best_classifier_name = final_classification["F1"].idxmax()
best_classifier_metrics = final_classification.loc[
    best_classifier_name
]

print("\n" + "=" * 70)
print("20. FINAL RECOMMENDATION")
print("=" * 70)

print(
    f"{best_classifier_name} has the highest F1 score among the "
    f"classifiers evaluated, with F1="
    f"{best_classifier_metrics['F1']:.4f}."
)

print(
    f"Its accuracy is {best_classifier_metrics['Accuracy']:.4f}, "
    f"precision is {best_classifier_metrics['Precision']:.4f}, "
    f"recall is {best_classifier_metrics['Recall']:.4f}, and "
    f"AUC is {best_classifier_metrics['AUC']:.4f}."
)

print(
    "These metrics indicate how well the classifier balances correct "
    "positive predictions, positive-case coverage, overall accuracy, "
    "and ranking performance."
)

print(
    "For deployment, the final saved pipeline below is the tuned "
    "Random Forest pipeline because it contains preprocessing and the "
    "Random Forest estimator in one end-to-end object."
)


# ============================================================
# 21. SAVE COMPLETE FITTED PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("21. SAVE COMPLETE PIPELINE")
print("=" * 70)

pipeline_path = BASE_DIR / "titanic_best_pipeline.joblib"

joblib.dump(
    best_rf_pipeline,
    pipeline_path
)

print(f"Saved: {pipeline_path}")


# ============================================================
# 22. RELOAD AND PREDICT RAW INPUT
# ============================================================

print("\n" + "=" * 70)
print("22. RELOAD SAVED PIPELINE AND TEST RAW INPUT")
print("=" * 70)

loaded_pipeline = joblib.load(
    pipeline_path
)

raw_sample = X_test.iloc[[0]]

loaded_prediction = loaded_pipeline.predict(
    raw_sample
)

loaded_probability = (
    loaded_pipeline
    .predict_proba(raw_sample)
)

print("\nRaw input:")
print(raw_sample.to_string(index=False))

print("\nActual target:")
print(int(y_test.iloc[0]))

print("\nReloaded pipeline prediction:")
print(int(loaded_prediction[0]))

print("\nPrediction probabilities:")
print(loaded_probability)


# ============================================================
# 23. FINAL ARTIFACT CHECK
# ============================================================

print("\n" + "=" * 70)
print("MODEL ARTIFACTS")
print("=" * 70)

for path in sorted(BASE_DIR.glob("*")):
    if path.is_file():
        print(path.name)

print("\nMODEL PIPELINE COMPLETE.")