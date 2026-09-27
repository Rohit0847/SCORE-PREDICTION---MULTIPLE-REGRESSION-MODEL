# ============================================================
# STUDENT PERFORMANCE PREDICTION - MACHINE LEARNING PROJECT
# Version 2
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

# Find the folder where project07.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CSV is expected to be in the same folder
csv_path = os.path.join(
    BASE_DIR,
    "student_performance_dataset.csv"
)

df = pd.read_csv(csv_path)


# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 4. REMOVE UNNECESSARY / LEAKAGE COLUMNS
# ============================================================

# student_id is only an identifier.
#
# final_grade is NOT used because it is directly related
# to the final exam score and could cause data leakage.

columns_to_remove = [
    "student_id",
    "final_grade"
]

df = df.drop(
    columns=columns_to_remove
)


# ============================================================
# 5. DEFINE FEATURES AND TARGET
# ============================================================

# Target variable
y = df["final_exam_score"]

# Remove target from input features
X = df.drop(
    columns=["final_exam_score"]
)


print("\n========================================")
print("FEATURES USED")
print("========================================")

print(X.columns.tolist())

print("\nTarget:")
print("final_exam_score")


# ============================================================
# 6. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\n========================================")
print("FEATURE TYPES")
print("========================================")

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n========================================")
print("TRAIN / TEST SPLIT")
print("========================================")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 8. PREPROCESSING
# ============================================================

# Numerical data:
# Missing values are replaced with the median.

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# Categorical data:
# Missing values are replaced with the most common value.
# Then categorical values are converted into numbers.

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# Combine numerical and categorical preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 9. CREATE ML MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


# ============================================================
# 10. TRAIN AND COMPARE MODELS
# ============================================================

results = []

trained_models = {}

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")


for model_name, model in models.items():

    print(
        f"\nTraining {model_name}..."
    )

    # Create complete pipeline
    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    # Train model
    pipeline.fit(
        X_train,
        y_train
    )

    # Save trained pipeline temporarily
    trained_models[model_name] = pipeline

    # --------------------------------------------------------
    # CROSS VALIDATION
    # --------------------------------------------------------

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="r2"
    )

    cv_r2 = cv_scores.mean()

    # --------------------------------------------------------
    # TEST SET PREDICTION
    # --------------------------------------------------------

    y_pred = pipeline.predict(
        X_test
    )

    # --------------------------------------------------------
    # TEST METRICS
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    results.append(
        {
            "Model": model_name,
            "CV R2": cv_r2,
            "MAE": mae,
            "RMSE": rmse,
            "Test R2": r2
        }
    )


# ============================================================
# 11. DISPLAY MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="CV R2",
    ascending=False
)

print("\n========================================")
print("MODEL RESULTS")
print("========================================")

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 12. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

print("\n========================================")
print("BEST MODEL")
print("========================================")

print(
    "Selected model:",
    best_model_name
)


# ============================================================
# 13. FINAL TEST PERFORMANCE
# ============================================================

best_predictions = best_model.predict(
    X_test
)

best_mae = mean_absolute_error(
    y_test,
    best_predictions
)

best_mse = mean_squared_error(
    y_test,
    best_predictions
)

best_rmse = np.sqrt(
    best_mse
)

best_r2 = r2_score(
    y_test,
    best_predictions
)


print("\n========================================")
print("BEST MODEL PERFORMANCE")
print("========================================")

print(f"MAE  : {best_mae:.2f}")
print(f"MSE  : {best_mse:.2f}")
print(f"RMSE : {best_rmse:.2f}")
print(f"R²   : {best_r2:.2f}")


# ============================================================
# 14. SAVE BEST MODEL
# ============================================================

model_path = os.path.join(
    BASE_DIR,
    "student_performance_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print("\n========================================")
print("MODEL SAVING")
print("========================================")

print(
    "Model saved successfully!"
)

print(
    "Location:",
    model_path
)

print(
    "Model exists:",
    os.path.exists(model_path)
)


# ============================================================
# 15. CUSTOM STUDENT PREDICTION
# ============================================================

print("\n========================================")
print("CUSTOM STUDENT PREDICTION")
print("========================================")

print("\nEnter information about the student.")

try:

    study_time = float(
        input(
            "Study time (hours): "
        )
    )

    attendance = float(
        input(
            "Attendance percentage: "
        )
    )

    sleep = float(
        input(
            "Sleep hours: "
        )
    )

    previous_grade = float(
        input(
            "Previous grade: "
        )
    )

    gender = input(
        "Gender (Male/Female): "
    )

    parental_education = input(
        "Parental education: "
    )

    internet_access = input(
        "Internet access (Yes/No): "
    )

    extracurricular = input(
        "Extracurricular activities (Yes/No): "
    )

    part_time_job = input(
        "Part-time job (Yes/No): "
    )


    # Create DataFrame for new student

    new_student = pd.DataFrame(
        {
            "gender": [gender],

            "study_time_hours": [
                study_time
            ],

            "attendance_percent": [
                attendance
            ],

            "sleep_hours": [
                sleep
            ],

            "parental_education": [
                parental_education
            ],

            "internet_access": [
                internet_access
            ],

            "extracurricular_activities": [
                extracurricular
            ],

            "part_time_job": [
                part_time_job
            ],

            "previous_grade": [
                previous_grade
            ]
        }
    )


    # Make prediction

    predicted_score = best_model.predict(
        new_student
    )


    print("\n========================================")

    print(
        f"Predicted Final Exam Score: "
        f"{predicted_score[0]:.2f}"
    )

    print("========================================")


except ValueError:

    print(
        "\nPlease enter valid numerical values."
    )


# ============================================================
# 16. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.7
)


# Perfect prediction line

minimum = min(
    y_test.min(),
    best_predictions.min()
)

maximum = max(
    y_test.max(),
    best_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)


plt.xlabel(
    "Actual Final Exam Score"
)

plt.ylabel(
    "Predicted Final Exam Score"
)

plt.title(
    f"Actual vs Predicted - {best_model_name}"
)

plt.grid(True)

plt.show()


# ============================================================
# 17. MODEL COMPARISON GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    results_df["Model"],
    results_df["Test R2"]
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "R² Score"
)

plt.title(
    "Model Performance Comparison"
)

plt.xticks(
    rotation=20
)

plt.grid(
    axis="y"
)

plt.show()


# ============================================================
# 18. FINAL MESSAGE
# ============================================================

print("\n========================================")
print("ML PROJECT COMPLETED SUCCESSFULLY!")
print("========================================")

print(
    f"Best model: {best_model_name}"
)

print(
    f"Best test R²: {best_r2:.2f}"
)

print(
    f"Best test MAE: {best_mae:.2f}"
)

print(
    f"Best test RMSE: {best_rmse:.2f}"
)

print(
    "\nSaved model:"
)

print(
    model_path
)