import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_excel("data/ppc_campaign_performance_data.xlsx")

print("Dataset shape:", df.shape)


# ==========================================
# 2. DATE FEATURE ENGINEERING
# ==========================================

df["Date"] = pd.to_datetime(df["Date"])

df["Month"] = df["Date"].dt.month
df["Quarter"] = df["Date"].dt.quarter


# ==========================================
# 3. SELECT FEATURES AND TARGET
# ==========================================

features = [
    "Budget",
    "Duration",
    "Platform",
    "Content_Type",
    "Target_Age",
    "Target_Gender",
    "Region",
    "Month",
    "Quarter"
]

target = "Conversion_Rate"

X = df[features]
y = df[target]


# ==========================================
# 4. DEFINE FEATURE TYPES
# ==========================================

categorical_features = [
    "Platform",
    "Content_Type",
    "Target_Age",
    "Target_Gender",
    "Region"
]

numerical_features = [
    "Budget",
    "Duration",
    "Month",
    "Quarter"
]


# ==========================================
# 5. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)


# ==========================================
# 6. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 7. DEFINE MODELS
# ==========================================

models = {

    "Ridge Regression": Ridge(),

    "Random Forest": RandomForestRegressor(
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}


# ==========================================
# 8. HYPERPARAMETER GRIDS
# ==========================================

param_grids = {

    "Ridge Regression": {
        "model__alpha": [0.1, 1, 10, 100]
    },

    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5]
    },

    "Gradient Boosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3]
    }
}


# ==========================================
# 9. MODEL TRAINING + TUNING
# ==========================================

results = {}

best_model = None
best_model_name = None
best_r2 = -float("inf")


for name, model in models.items():

    print("\n================================")
    print("Training:", name)
    print("================================")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    grid_search = GridSearchCV(
        pipeline,
        param_grids[name],
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    predictions = grid_search.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    print("Best parameters:", grid_search.best_params_)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    if r2 > best_r2:

        best_r2 = r2
        best_model = grid_search.best_estimator_
        best_model_name = name


# ==========================================
# 10. FINAL MODEL COMPARISON
# ==========================================

print("\n\n================================")
print("FINAL MODEL COMPARISON")
print("================================")

for name, metrics in results.items():

    print(f"\n{name}")

    print(
        f"MAE  : {metrics['MAE']:.4f}"
    )

    print(
        f"RMSE : {metrics['RMSE']:.4f}"
    )

    print(
        f"R²   : {metrics['R2']:.4f}"
    )


# ==========================================
# 11. BEST MODEL
# ==========================================

print("\n================================")
print("BEST MODEL")
print("================================")

print("Model:", best_model_name)
print(f"R² Score: {best_r2:.4f}")


# ==========================================
# 12. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    "models/best_conversion_model.pkl"
)

print("\nBest model saved successfully!")
print("Location: models/best_conversion_model.pkl")