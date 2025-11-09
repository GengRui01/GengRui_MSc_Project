import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

from datetime import datetime
from utils.db_connector import fetch_student_data
from utils.model_trainer import evaluate_model, infer_one


# -------------------------------
# Streamlit page configuration
# -------------------------------
st.set_page_config(
    page_title="Student Behaviour Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Load trained model and scaler
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/trained_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None, None

model, scaler = load_model()

# -------------------------------
# Data Fetching
# -------------------------------
@st.cache_data(ttl=600)
def load_data():
    """Fetch data from database and cache it for 10 minutes."""
    try:
        df = fetch_student_data()
        st.sidebar.success("✅ Data loaded successfully (cached)!")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.sidebar.success(f"🕒 Last refreshed: *{timestamp}*")
        return df
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        st.stop()

# -------------------------------
# Sidebar: Controls Panel
# -------------------------------
st.sidebar.header("🔍 Controls Panel")
st.sidebar.write("Use this panel to refresh and manage the student behaviour dataset.")

refresh = st.sidebar.button("🔄 Refresh Data")

if refresh:
    # Clear both data and model cache
    st.cache_data.clear()
    st.sidebar.success("🔁 Data cache cleared! Reloading new data...")
    df = load_data()
    st.rerun()
else:
    df = load_data()

# -------------------------------
# Model Prediction Section
# -------------------------------
st.title("🎓 Student Behaviour Analysis and Prediction Dashboard")
st.markdown("This dashboard displays the results of machine-learning analysis of student behaviour on online learning platforms, including engagement and performance predictions, and highlights at-risk learners to support data-driven teaching.")

st.markdown("---")

st.subheader("🎯 Student Risk Prediction")

# Display the model’s predicted high-risk students for quick review
st.markdown("#### 📋 Predicted High-Risk Students")

X = df[["login_count", "time_spent", "quiz_attempts"]]
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)
df["risk_level"] = ["High Risk" if p == 0 else "Low Risk" for p in y_pred]

high_risk = df[df["risk_level"] == "High Risk"][["student_id", "login_count", "time_spent", "quiz_attempts", "completion_rate"]]
st.dataframe(high_risk, use_container_width=True)

# Provide dynamic filtering controls for focused analysis
st.markdown("#### 🧩 Interactive Filters")

# Select a student to view detailed engagement data and risk level predictions.
selected_student = st.selectbox("Select a student:", options=df["student_id"].unique())
filtered_df = df[df["student_id"] == selected_student]
# Display selected student's data
st.write(f"Showing detailed engagement data for **{selected_student}**:")
st.dataframe(filtered_df, use_container_width=True)

st.markdown("---")

# -------------------------------
# Visualisation Section
# -------------------------------
st.subheader("🎨 Multi-dimensional Visualisation")

# Line chart: time spent vs completion rate
st.write("#### 🔵 Time Spent vs Completion Rate")
# st.line_chart(df[["time_spent", "completion_rate"]])
fig = px.scatter(
    df,
    x="time_spent",
    y="completion_rate",
    color="risk_level",
    trendline="ols",  # Automatic linear fitting
    color_discrete_map={"High Risk": "#E74C3C", "Low Risk": "#27AE60"},
    labels={
        "time_spent": "Time Spent (hours)",
        "completion_rate": "Completion Rate"
    }
)

fig.update_traces(marker_size=9, opacity=0.7)
st.plotly_chart(fig, use_container_width=True)

# Risk and Engagement Overview (Side-by-Side)
col1, col2 = st.columns(2)

# Left side：Bar chart: distribution of risk levels
with col1:
    st.write("#### 🍩 Risk Level Distribution")
    fig = px.pie(
        df,
        names="risk_level",
        color="risk_level",
        color_discrete_map={"High Risk": "#E74C3C", "Low Risk": "#27AE60"},
        hole=0.4,
    )
    fig.update_traces(
        textinfo="percent+label",
        texttemplate="%{label}<br>%{percent}",
        hovertemplate="<b>%{label}</b><br>Students: %{value}<br>Percentage: %{percent}<extra></extra>",
        pull=[0.1, 0]
    )
    st.plotly_chart(fig, use_container_width=True)

# Right side：Bar chart: average engagement comparison by risk level
with col2:
    st.write("#### 📊 Engagement Pattern Comparison")
    # Calculate the average value for each risk level
    avg_by_risk = df.groupby("risk_level")[["login_count", "time_spent", "quiz_attempts", "completion_rate"]].mean().reset_index()
    # Convert to long format for easier plotting
    avg_melted = avg_by_risk.melt(id_vars="risk_level", var_name="Metric", value_name="Average Value")
    # Plot the grouped bar chart
    fig_bar = px.bar(
        avg_melted,
        x="Metric",
        y="Average Value",
        color="risk_level",
        barmode="group",
        color_discrete_map={"High Risk": "#E74C3C", "Low Risk": "#27AE60"},
        text_auto=".2f"
    )
    # Improve the layout
    fig_bar.update_layout(
        xaxis_title="Metric",
        yaxis_title="Avg Value",
        showlegend=False,
        bargap=0.25,
        margin=dict(t=40, b=0, l=0, r=0),
        template="plotly_white",
        font=dict(size=12),
        height=400
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# -------------------------------
# 🧑‍🎓 Single Student Prediction
# -------------------------------
st.markdown("---")
st.subheader("🧑‍🎓 Single Student Prediction")

with st.form("single_predict_form", clear_on_submit=False):
    # Identity (display only, not used in model input)
    c0, c1 = st.columns([1, 1])
    with c0: student_name = st.text_input("Student Name", value="")
    with c1: student_id = st.text_input("Student ID", value="")
    # Core features (must match original training features)
    c2, c3, c4 = st.columns([1, 1, 1])
    with c2: login_count = st.number_input("Login Count", min_value=0, step=1, value=5)
    with c3: time_spent = st.number_input("Study Time (hours)", min_value=0.0, step=0.5, value=3.0, format="%.2f")
    with c4: quiz_attempts = st.number_input("Quiz Attempts", min_value=0, step=1, value=2)

    if st.form_submit_button("Predict"):
        try:
            # Assemble features in the correct order
            features = {
                "login_count": int(login_count),
                "time_spent": float(time_spent),
                "quiz_attempts": int(quiz_attempts),
            }

            # Run inference
            result = infer_one(features)
            prob = float(result.get("risk_probability", 0.0))
            risk_level = result.get("risk_category", "Unknown");
            prob_pct = f"{prob*100:.2f}%"

            # Simple rule-based recommendation
            if risk_level == "High Risk":
                recommendation = "This student's learning progress should be closely monitored, paying attention to assignment completion rate and interaction frequency."
                card_renderer = st.error
            elif risk_level == "Medium Risk":
                recommendation = "This student's learning performance is relatively stable; it is recommended to appropriately increase learning engagement and the frequency of quizzes."
                card_renderer = st.warning
            else:
                recommendation = "This student's learning situation is good; please maintain the current learning pace."
                card_renderer = st.success

            # Render card
            st.write("##### Result")
            if student_name or student_id:
                st.write(f"**Student:** {student_name or '—'}  |  **ID:** {student_id or '—'}")
            card_renderer(f"**{risk_level}**  •  Probability: **{prob_pct}**  \n📝 {recommendation}")

            with st.expander("View input features"):
                st.json(features)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------------------
# Model Evaluation Section
# -------------------------------
st.markdown("---")
st.subheader("🧠 Model Evaluation Results")

if st.button("Evaluate Model"):
    try:
        # Run model evaluation
        results = evaluate_model()

        # ---- RQ1: Data realism ----
        st.write("#### 📚 Synthetic Data Realism")
        st.write("Descriptive Statistics:")
        st.dataframe(results["desc"], use_container_width=True)

        st.write("Correlation Heatmap:")
        fig_corr = px.imshow(results["corr"], text_auto=False, color_continuous_scale="RdBu_r", aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)

        # ---- RQ2: Predictive performance ----
        st.write("#### ⚖️ Predictive Performance (Positive class = High-risk)")
        st.write("Model Metrics:")
        st.dataframe(pd.DataFrame([results["metrics"]]), use_container_width=True)

        st.write("Confusion Matrix:")
        cm = results["confusion_matrix"]
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           labels=dict(x="Predicted", y="Actual", color="Count"))
        st.plotly_chart(fig_cm, use_container_width=True)

        st.write("Classification Report Summary:")
        st.write({"Report": results["report"]})

        # ---- RQ3: Visualisation & Interpretability ----
        st.write("#### 🧾 Visualisation and Interpretability")
        st.write("SHAP Beeswarm:")
        st.image("models/shap_explain.png", use_column_width=True)
        st.markdown("""
        - The dashboard provides an **integrated evaluation workflow**, integrating data realism and performance results.   
        - Educators can visually verify model effectiveness and inspect synthetic data distributions.  
        - Such transparency supports explainability and evidence-based intervention decisions.
        """)
    except Exception as e:
        st.error(f"❌ Evaluation failed: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("© 2025 Geng Rui — MSc Project | Student Behaviour Analysis and Prediction on Online Learning Platforms Based on Machine Learning")