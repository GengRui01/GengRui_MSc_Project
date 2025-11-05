# 🎓 Student Behavior Analysis and Prediction Dashboard

This project provides a persisted ML model + StandardScaler saved via joblib and a Streamlit dashboard backed by MySQL (with synthetic auto-seeding on first run) . 

It uses cached data/model loading for responsiveness and reproducibility (fixed random seed), and renders interactive charts with Plotly. 

The app supports dataset-level risk screening and Single Student Prediction, and includes a built-in model evaluation workflow covering data realism, predictive performance, and visual interpretability.

---

## ✨ Features

### 1. Backend Capabilities (Data / Model / Automation & Reproducibility)

Data layer: connects to MySQL; when the target table is empty (e.g. the first query) it auto-seeds synthetic data (fixed random seed for reproducibility).

Model layer: a scikit-learn–based model developed and trained in this project, then persisted via joblib for deployment. After evaluating Logistic Regression (baseline) and Gradient Boosting, Random Forest was chosen as the production classifier. The task is binary risk prediction (High/Low), and evaluation uses the same setting.

### 2. Controls Panel

One-click **Refresh Data**: clears `st.cache_data`, reloads the dataset, and triggers `st.rerun()` to update all views.

Caching: data uses `@st.cache_data(ttl=600)` (10-min expiry).

Status: shows a success toast and a “Last refreshed” timestamp in the sidebar.

Safety: no schema changes; Model/scaler use `@st.cache_resource` and are not affected.

### 3. Student Risk Prediction (dataset level)

Predicted High-Risk Students: table of high-risk learners based on current data (`login_count`, `time_spent`, `quiz_attempts`, `completion_rate`).

Interactive Filters: select by `student_id` to view that student’s detailed engagement record.

### 4. Multi-dimensional Visualization (engagement & risk overview)

Time Spent vs Completion Rate: scatter with OLS trendline, colored by `risk_level`.

Risk Level Distribution: donut chart (High / Low).

Engagement Pattern Comparison: grouped bars comparing mean `login_count`, `time_spent`, `quiz_attempts`, `completion_rate` by `risk_level`.

### 5. Single Student Prediction (one-sample inference)

Input: `login_count`, `time_spent`, `quiz_attempts` (name/ID are for display only).

Click **Predict** to run inference for a single student record.

Outputs a risk category (High / Medium / Low) and probability with a short rule-based recommendation; expandable panel shows the input features (JSON).

### 6. Model Evaluation Results (Positive class = High-risk)

Click **Evaluate Model** to run the evaluation workflow covering RQ1–RQ3.

RQ1 — Data realism: Outputs `describe` and `corr`; check distributions, feature correlations, and class balance.

RQ2 — Predictive performance: Outputs Metrics (Acc/Prec/Rec/F1/AUC), Confusion Matrix, and Classification Report.  
Positive class = High-risk (class 0); threshold = 0.5. Focus on FN (missed high-risk).

RQ3 — Interpretability: A SHAP summary was already stored as `models/shap_force_summary.html` when `train_model`, shown it under Visualization and Interpretability. Positive SHAP pushes toward High-risk; negative toward Low-risk.

---

## 📂 Project Structure

```
GengRui_MSc_Project/
│
├─ db/
│  └─ init.sql                  # Creates the MySQL schema and tables used by this project.
│
├─ models/
│  ├─ scaler.pkl                # A fitted StandardScaler that normalises features exactly as in training.
│  ├─ shap_force_summary.html   # SHAP summary saved during evaluation and displayed in Visualization and Interpretability.
│  └─ trained_model.pkl         # The trained machine-learning model used by the dashboard.
│
├─ utils/
│  ├─ data_generator.py         # Generates synthetic student-behaviour data and writes it into the database.
│  ├─ db_connector.py           # Creates an SQL connection and fetches student records, auto-seeding the table if it’s empty.
│  └─ model_trainer.py          # Holds the logic for model training, evaluation, and single-record inference.
│
├─ .gitignore                   # Ignore rules for caches, editor files, and other local artefacts.
├─ app.py                       # The Streamlit dashboard that brings prediction, visualisation, and evaluation together.
└─ README.md                    # Project documentation.
```

---

## 🚀 How to Run

### 1. Environment Setup

To set up the environment, install all required Python packages with a single command:  

```bash
pip install streamlit pandas numpy scikit-learn sqlalchemy pymysql plotly joblib shap
```

### 2. Initialize Database

Run the SQL script once to create the database and tables:

```bash
mysql -u root -p < db/init.sql
```

> *Note:*  
>`init.sql` only creates the database and tables.  
>On the first query, if the table is empty, the app will auto-seed synthetic data via `utils/data_generator.py`.

### 3. Configure connection (Once)

Open `utils/db_connector.py` and set MySQL username and password in SQLAlchemy URL, e.g.

```python
engine = create_engine("mysql+pymysql://<USER>:<PASSWORD>@localhost:3306/gengrui_msc")
```

### 4. Train Model (Optional)

The pre-trained model (`trained_model.pkl`) and scaler (`scaler.pkl`) and shap (`shap_force_summary.html`) are already included under the `models/` directory.

If you wish to retrain the model, you can manually run:

```bash
python utils/model_trainer.py
```

This will update the model files under the `models/` directory.

### 5. Launch Dashboard

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501).

---

## 🧑‍🎓 Author

**Geng Rui (耿锐)**  

MSc Project – *Student Behavior Analysis and Prediction on Online Learning Platforms Based on Machine Learning*  

Supervisor: *SURYANTO*
