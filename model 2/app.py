# ============================================================
# STUDENT PERFORMANCE PREDICTION
# MULTIPLE LINEAR REGRESSION - STREAMLIT APP
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        background-color: #f7f9fc;
        border: 1px solid #e5e7eb;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. TITLE
# ============================================================

st.markdown(
    '<div class="title">🎓 Student Performance Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Multiple Linear Regression Machine Learning Model'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# 4. LOAD DATASET
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

csv_path = os.path.join(
    BASE_DIR,
    "student_performance_dataset.csv"
)


if not os.path.exists(csv_path):

    st.error(
        "❌ Dataset not found.\n\n"
        "Please keep 'student_performance_dataset.csv' "
        "in the same folder as app.py."
    )

    st.stop()


df_original = pd.read_csv(csv_path)


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🎓 Student Performance")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 Dataset",
        "📈 Model Performance",
        "🎯 Predict Score",
        "📉 Visualizations"
    ]
)


# ============================================================
# 6. DATA PREPARATION
# ============================================================

df = df_original.copy()


# Remove identifier and leakage column

columns_to_remove = [
    "student_id",
    "final_grade"
]


existing_columns = [
    column
    for column in columns_to_remove
    if column in df.columns
]


df = df.drop(
    columns=existing_columns
)


# ============================================================
# 7. DEFINE TARGET AND FEATURES
# ============================================================

if "final_exam_score" not in df.columns:

    st.error(
        "❌ 'final_exam_score' column was not found "
        "in the dataset."
    )

    st.stop()


# Target variable

y = df["final_exam_score"]


# Independent variables

X = df.drop(
    columns=["final_exam_score"]
)


# ============================================================
# 8. IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 10. NUMERICAL PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


# ============================================================
# 11. CATEGORICAL PREPROCESSING
# ============================================================

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
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


# ============================================================
# 12. COMBINE PREPROCESSING
# ============================================================

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
# 13. MULTIPLE LINEAR REGRESSION MODEL
# ============================================================

linear_regression = LinearRegression()


# Complete ML pipeline

model = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),

        (
            "regression",
            linear_regression
        )
    ]
)


# ============================================================
# 14. TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(
    X_train,
    y_train
):

    regression_model = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessor
            ),

            (
                "regression",
                LinearRegression()
            )
        ]
    )

    regression_model.fit(
        X_train,
        y_train
    )

    return regression_model


with st.spinner(
    "🤖 Training Multiple Linear Regression model..."
):

    model = train_model(
        X_train,
        y_train
    )


# ============================================================
# 15. TEST PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 16. EVALUATION METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)


mse = mean_squared_error(
    y_test,
    y_pred
)


rmse = np.sqrt(
    mse
)


r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# 17. CROSS VALIDATION
# ============================================================

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="r2"
)


cv_r2 = cv_scores.mean()


# ============================================================
# 18. SAVE MODEL
# ============================================================

model_path = os.path.join(
    BASE_DIR,
    "multiple_linear_regression_model.pkl"
)


joblib.dump(
    model,
    model_path
)


# ============================================================
# 19. DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header(
        "📊 Student Performance Dashboard"
    )

    st.write(
        "This application predicts a student's "
        "final examination score using "
        "**Multiple Linear Regression**."
    )

    st.markdown("---")


    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "👨‍🎓 Students",
            len(df_original)
        )


    with col2:

        st.metric(
            "📋 Features",
            len(X.columns)
        )


    with col3:

        st.metric(
            "📈 R² Score",
            f"{r2:.3f}"
        )


    with col4:

        st.metric(
            "📉 RMSE",
            f"{rmse:.2f}"
        )


    st.markdown("---")


    st.subheader(
        "📌 Model Information"
    )


    model_information = pd.DataFrame(
        {
            "Parameter": [
                "Algorithm",
                "Problem Type",
                "Target Variable",
                "Training Samples",
                "Testing Samples",
                "Cross-Validation"
            ],

            "Value": [
                "Multiple Linear Regression",
                "Regression",
                "final_exam_score",
                len(X_train),
                len(X_test),
                "5-Fold Cross Validation"
            ]
        }
    )


    st.dataframe(
        model_information,
        use_container_width=True,
        hide_index=True
    )


    st.markdown("---")


    st.subheader(
        "🎯 Model Performance"
    )


    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)


    with metric_col1:

        st.metric(
            "MAE",
            f"{mae:.2f}"
        )


    with metric_col2:

        st.metric(
            "MSE",
            f"{mse:.2f}"
        )


    with metric_col3:

        st.metric(
            "RMSE",
            f"{rmse:.2f}"
        )


    with metric_col4:

        st.metric(
            "Test R²",
            f"{r2:.3f}"
        )


# ============================================================
# 20. DATASET PAGE
# ============================================================

elif page == "📊 Dataset":

    st.header(
        "📊 Dataset Explorer"
    )


    # Dataset statistics

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Rows",
            df_original.shape[0]
        )


    with col2:

        st.metric(
            "Columns",
            df_original.shape[1]
        )


    with col3:

        st.metric(
            "Missing Values",
            int(
                df_original.isnull()
                .sum()
                .sum()
            )
        )


    with col4:

        st.metric(
            "Duplicate Rows",
            int(
                df_original.duplicated()
                .sum()
            )
        )


    st.markdown("---")


    # Dataset preview

    st.subheader(
        "🔍 Dataset Preview"
    )


    st.dataframe(
        df_original.head(10),
        use_container_width=True
    )


    st.markdown("---")


    # Columns

    st.subheader(
        "📋 Dataset Columns"
    )


    column_information = pd.DataFrame(
        {
            "Column":
                df_original.columns,

            "Data Type":
                [
                    str(
                        df_original[column].dtype
                    )
                    for column in df_original.columns
                ],

            "Missing Values":
                [
                    int(
                        df_original[column]
                        .isnull()
                        .sum()
                    )
                    for column in df_original.columns
                ],

            "Unique Values":
                [
                    int(
                        df_original[column]
                        .nunique()
                    )
                    for column in df_original.columns
                ]
        }
    )


    st.dataframe(
        column_information,
        use_container_width=True,
        hide_index=True
    )


    st.markdown("---")


    st.subheader(
        "🎯 Target Variable"
    )


    st.info(
        "Target variable: **final_exam_score**"
    )


# ============================================================
# 21. MODEL PERFORMANCE PAGE
# ============================================================

elif page == "📈 Model Performance":

    st.header(
        "📈 Multiple Linear Regression Performance"
    )


    st.write(
        "The model is evaluated using standard "
        "regression performance metrics."
    )


    # Metrics

    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Training / Validation"
        )

        st.metric(
            "5-Fold Cross Validation R²",
            f"{cv_r2:.4f}"
        )


    with col2:

        st.subheader(
            "Testing"
        )

        st.metric(
            "Test R²",
            f"{r2:.4f}"
        )


    st.markdown("---")


    # Metrics table

    st.subheader(
        "📊 Evaluation Metrics"
    )


    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Mean Absolute Error (MAE)",
                "Mean Squared Error (MSE)",
                "Root Mean Squared Error (RMSE)",
                "R² Score",
                "Cross-Validation R²"
            ],

            "Value": [
                round(mae, 4),
                round(mse, 4),
                round(rmse, 4),
                round(r2, 4),
                round(cv_r2, 4)
            ]
        }
    )


    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )


    st.markdown("---")


    # Explanation

    st.subheader(
        "📌 Regression Metrics"
    )


    st.markdown(
        """
        **MAE:** Average absolute difference between
        actual and predicted scores.

        **MSE:** Average squared difference between
        actual and predicted scores.

        **RMSE:** Square root of MSE. It represents
        prediction error in the same unit as the target.

        **R² Score:** Measures how much variation in
        the target variable is explained by the model.

        **Cross-Validation R²:** Average R² obtained
        using 5-fold cross-validation.
        """
    )


# ============================================================
# 22. PREDICTION PAGE
# ============================================================

elif page == "🎯 Predict Score":

    st.header(
        "🎯 Predict Final Exam Score"
    )


    st.write(
        "Enter the student's information "
        "to generate a predicted final exam score."
    )


    st.markdown("---")


    with st.form(
        "student_prediction_form"
    ):

        st.subheader(
            "📚 Academic Information"
        )


        col1, col2 = st.columns(2)


        # Study Time

        with col1:

            study_time = st.number_input(
                "📚 Study Time (hours)",
                min_value=0.0,
                max_value=24.0,
                value=5.0,
                step=0.5
            )


        # Attendance

        with col2:

            attendance = st.number_input(
                "📅 Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0
            )


        # Sleep

        with col1:

            sleep = st.number_input(
                "😴 Sleep Hours",
                min_value=0.0,
                max_value=24.0,
                value=7.0,
                step=0.5
            )


        # Previous Grade

        with col2:

            previous_grade = st.number_input(
                "📈 Previous Grade",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                step=1.0
            )


        st.markdown("---")


        st.subheader(
            "👤 Student Information"
        )


        # Gender

        if "gender" in categorical_features:

            gender_values = (
                df_original["gender"]
                .dropna()
                .unique()
                .tolist()
            )

            gender = st.selectbox(
                "Gender",
                gender_values
            )

        else:

            gender = None


        # Parental Education

        if "parental_education" in categorical_features:

            parental_values = (
                df_original[
                    "parental_education"
                ]
                .dropna()
                .unique()
                .tolist()
            )

            parental_education = st.selectbox(
                "Parental Education",
                parental_values
            )

        else:

            parental_education = None


        # Internet Access

        if "internet_access" in categorical_features:

            internet_values = (
                df_original[
                    "internet_access"
                ]
                .dropna()
                .unique()
                .tolist()
            )

            internet_access = st.selectbox(
                "Internet Access",
                internet_values
            )

        else:

            internet_access = None


        # Extracurricular

        if "extracurricular_activities" in categorical_features:

            extracurricular_values = (
                df_original[
                    "extracurricular_activities"
                ]
                .dropna()
                .unique()
                .tolist()
            )

            extracurricular = st.selectbox(
                "Extracurricular Activities",
                extracurricular_values
            )

        else:

            extracurricular = None


        # Part Time Job

        if "part_time_job" in categorical_features:

            part_time_values = (
                df_original[
                    "part_time_job"
                ]
                .dropna()
                .unique()
                .tolist()
            )

            part_time_job = st.selectbox(
                "Part-Time Job",
                part_time_values
            )

        else:

            part_time_job = None


        st.markdown("---")


        submitted = st.form_submit_button(
            "🚀 Predict Final Exam Score"
        )


    # ========================================================
    # MAKE PREDICTION
    # ========================================================

    if submitted:

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


        # Match exact training columns

        new_student = new_student[
            X.columns
        ]


        # Predict

        predicted_score = model.predict(
            new_student
        )[0]


        # Keep prediction between 0 and 100

        predicted_score = max(
            0,
            min(
                100,
                predicted_score
            )
        )


        st.markdown(
            '<div class="prediction-box">',
            unsafe_allow_html=True
        )


        st.subheader(
            "🎯 Predicted Final Exam Score"
        )


        st.markdown(
            f"# {predicted_score:.2f} / 100"
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


        # Performance message

        if predicted_score >= 90:

            st.success(
                "🌟 Excellent predicted performance!"
            )

        elif predicted_score >= 75:

            st.success(
                "👏 Good predicted performance!"
            )

        elif predicted_score >= 60:

            st.info(
                "📚 Average predicted performance."
            )

        else:

            st.warning(
                "⚠️ The predicted score indicates "
                "that additional preparation may help."
            )


        st.markdown("---")


        st.subheader(
            "📋 Student Input"
        )


        st.dataframe(
            new_student,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 23. VISUALIZATIONS
# ============================================================

elif page == "📉 Visualizations":

    st.header(
        "📉 Model Visualizations"
    )


    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    st.subheader(
        "Actual vs Predicted Scores"
    )


    fig1, ax1 = plt.subplots(
        figsize=(9, 6)
    )


    ax1.scatter(
        y_test,
        y_pred,
        alpha=0.7
    )


    minimum = min(
        y_test.min(),
        y_pred.min()
    )


    maximum = max(
        y_test.max(),
        y_pred.max()
    )


    # Perfect prediction line

    ax1.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )


    ax1.set_xlabel(
        "Actual Final Exam Score"
    )


    ax1.set_ylabel(
        "Predicted Final Exam Score"
    )


    ax1.set_title(
        "Actual vs Predicted Scores"
    )


    ax1.grid(True)


    st.pyplot(fig1)


    st.markdown("---")


    # ========================================================
    # RESIDUAL PLOT
    # ========================================================

    st.subheader(
        "Residual Plot"
    )


    residuals = y_test - y_pred


    fig2, ax2 = plt.subplots(
        figsize=(9, 6)
    )


    ax2.scatter(
        y_pred,
        residuals,
        alpha=0.7
    )


    ax2.axhline(
        y=0,
        linestyle="--"
    )


    ax2.set_xlabel(
        "Predicted Score"
    )


    ax2.set_ylabel(
        "Residual"
    )


    ax2.set_title(
        "Residual Analysis"
    )


    ax2.grid(True)


    st.pyplot(fig2)


    st.markdown("---")


    # ========================================================
    # SCORE DISTRIBUTION
    # ========================================================

    st.subheader(
        "Final Exam Score Distribution"
    )


    fig3, ax3 = plt.subplots(
        figsize=(9, 5)
    )


    ax3.hist(
        df_original[
            "final_exam_score"
        ],
        bins=20
    )


    ax3.set_xlabel(
        "Final Exam Score"
    )


    ax3.set_ylabel(
        "Number of Students"
    )


    ax3.set_title(
        "Distribution of Final Exam Scores"
    )


    ax3.grid(
        axis="y"
    )


    st.pyplot(fig3)


# ============================================================
# 24. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Student Performance Prediction | "
    "Multiple Linear Regression"
)