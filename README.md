# SCORE-PREDICTION---MULTIPLE-REGRESSION-MODEL

🎓 **Student Performance Prediction** — a machine learning project that predicts a student's **final exam score** from academic and lifestyle factors such as study time, attendance, sleep, and family background. Includes a full model-training/comparison script and an interactive **Streamlit** web app for exploring the data and making predictions.

🔗 Repo: [Rohit0847/SCORE-PREDICTION---MULTIPLE-REGRESSION-MODEL](https://github.com/Rohit0847/SCORE-PREDICTION---MULTIPLE-REGRESSION-MODEL)

---

## 📌 Overview

This project trains regression models on a student performance dataset and serves the best-performing model through a Streamlit dashboard. Users can input a student's details (study habits, attendance, sleep, parental education, etc.) and get a predicted final exam score, along with dataset exploration and model diagnostic visualizations.

## ✨ Features

- **Data preprocessing pipeline** — median imputation for numeric features, most-frequent imputation + one-hot encoding for categorical features, combined via `ColumnTransformer`
- **Model comparison** — trains and evaluates Linear Regression, Random Forest, and Gradient Boosting, selecting the best model by cross-validated R²
- **Streamlit app** with multiple pages:
  - 🏠 **Dashboard** — project summary
  - 📊 **Dataset** — raw data preview and exploration
  - 📈 **Model Performance** — MAE, RMSE, R², and cross-validation scores
  - 🎯 **Predict Score** — interactive form to predict a new student's final exam score
  - 📉 **Visualizations** — actual vs. predicted scatter plot, residual plot, and score distribution
- **Leakage-safe** — drops `student_id` (identifier) and `final_grade` (derived from the target) before training

## 🗂️ Project Structure

> ⚠️ In the current repo, the project files live inside the `model 2/` folder. Update the commands below (or move the files to the repo root) so the paths match your actual layout.

```
.
└── model 2/
    ├── app.py                                  # Streamlit web application
    ├── Untitled-1.py                           # Model training & comparison script (CLI)
    ├── student_performance_dataset.csv         # Dataset
    ├── student_performance_model.pkl           # Best model (from Untitled-1.py comparison)
    └── multiple_linear_regression_model.pkl    # Linear Regression model (from app.py)
└── README.md
```

## 📊 Dataset

The dataset (`student_performance_dataset.csv`) contains 1,000 student records with the following columns:

| Column | Description |
|---|---|
| `student_id` | Unique identifier (dropped before training) |
| `gender` | Student gender |
| `study_time_hours` | Hours spent studying |
| `attendance_percent` | Attendance percentage |
| `sleep_hours` | Average hours of sleep |
| `parental_education` | Parent's education level |
| `internet_access` | Whether the student has internet access |
| `extracurricular_activities` | Participation in extracurriculars |
| `part_time_job` | Whether the student has a part-time job |
| `previous_grade` | Previous academic grade |
| `final_exam_score` | **Target variable** — final exam score (0–100) |
| `final_grade` | Letter grade derived from the score (dropped — data leakage) |

## 🛠️ Tech Stack

- Python
- pandas, NumPy
- scikit-learn (pipelines, `ColumnTransformer`, Linear Regression, Random Forest, Gradient Boosting)
- Matplotlib
- Streamlit
- joblib (model persistence)

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Rohit0847/SCORE-PREDICTION---MULTIPLE-REGRESSION-MODEL.git
cd "SCORE-PREDICTION---MULTIPLE-REGRESSION-MODEL/model 2"
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn streamlit joblib
```

*(Optional: create a `requirements.txt` with the above packages and run `pip install -r requirements.txt`.)*

### 3. Train the model (optional — a pre-trained `.pkl` is included)

Runs a full comparison across Linear Regression, Random Forest, and Gradient Boosting, then saves the best model and prompts for a sample prediction:

```bash
python Untitled-1.py
```

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## 📈 Model Evaluation

Models are compared using 5-fold cross-validated R² on the training set, then evaluated on a held-out 20% test set using:

- **MAE** — Mean Absolute Error
- **MSE / RMSE** — Mean/Root Mean Squared Error
- **R²** — Coefficient of determination

## 🔮 Making a Prediction

In the Streamlit app's **Predict Score** page, enter:

- Study time (hours), attendance %, sleep hours, previous grade
- Gender, parental education, internet access, extracurricular activities, part-time job

...and the app returns a predicted final exam score (clipped to 0–100) with a performance summary.

## 📝 Notes

- `final_grade` is intentionally excluded from training since it is directly derived from `final_exam_score` and would leak the answer.
- `Untitled-1.py` compares three models and saves the best one as `student_performance_model.pkl`.
- `app.py` trains a dedicated Multiple Linear Regression model on launch and saves it as `multiple_linear_regression_model.pkl`.

## 📄 License

Add a license of your choice (e.g., MIT) here.
