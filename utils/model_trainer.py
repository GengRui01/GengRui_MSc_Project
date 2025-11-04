import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from utils.db_connector import fetch_student_data


def prepare_data(df):
    """
    Prepare features and labels for model training.
    Automatically adjusts threshold if label distribution is single-class.
    """
    print("[INFO] Preparing features and labels...")

    # Define input features
    X = df[["login_count", "time_spent", "quiz_attempts"]]

    # Initial threshold
    threshold = 0.7
    y = (df["completion_rate"] >= threshold).astype(int)

    # Check label distribution
    label_counts = y.value_counts()
    print(f"[DEBUG] Label distribution before adjustment:\n{label_counts}")

    # Auto-adjust threshold if dataset is single-class
    if len(label_counts) < 2:
        threshold = df["completion_rate"].median()
        y = (df["completion_rate"] >= threshold).astype(int)
        print(f"[WARN] Only one label class found. Adjusted threshold to median: {threshold:.2f}")
        print(f"[DEBUG] New label distribution:\n{y.value_counts()}")

    print("[INFO] Data preparation completed.")
    return X, y


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """
    Train multiple models and evaluate performance.
    Returns the best-performing model and result summary.
    """
    results = []

    for name, model in models.items():
        print(f"[INFO] Training model: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })

        print(f"[RESULT] {name}: Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

    # Select best model by F1-score
    results_df = pd.DataFrame(results)
    best_model_name = results_df.sort_values(by="F1-Score", ascending=False).iloc[0]["Model"]
    best_model = models[best_model_name]
    print(f"[INFO] Best model selected: {best_model_name}")

    return best_model, results_df


def explain_model(model, X_train, X_test):
    """
    Generate SHAP explainability visualization for the trained model.
    """
    print("[INFO] Generating SHAP explainability plot...")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("Feature Importance via SHAP Values")
    plt.tight_layout()
    plt.show()
    print("[INFO] SHAP explanation completed.")


def save_model(model, scaler, path_model="models/trained_model.pkl", path_scaler="models/scaler.pkl"):
    """
    Save model and scaler objects for later use in Streamlit.
    """
    joblib.dump(model, path_model)
    joblib.dump(scaler, path_scaler)
    print(f"[INFO] Model saved to {path_model}")
    print(f"[INFO] Scaler saved to {path_scaler}")


def train_model():
    """
    Main training pipeline that orchestrates data loading, training, evaluation, and saving.
    """
    print("[INFO] Loading data from database...")
    df = fetch_student_data()
    print(f"[INFO] Loaded {len(df)} records.")

    # Prepare dataset
    X, y = prepare_data(df)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    # Train and evaluate models
    best_model, results_df = train_and_evaluate(models, X_train_scaled, X_test_scaled, y_train, y_test)

    # Explain the best model
    explain_model(best_model, X_train_scaled, X_test_scaled)

    # Save model and scaler
    save_model(best_model, scaler)

    print("\n[SUMMARY] Model training completed successfully.")
    print(results_df)


if __name__ == "__main__":
    train_model()


def infer_one(feature_dict: dict):
    """
    Run inference for a single student record.
    Accepts a dictionary of feature values,
    applies the same preprocessing/scaling as training,
    and returns the predicted risk label and probability.
    """
    print("[INFO] Starting single-record inference ")

    # ----- Step 1: Load trained model and scaler -----
    model = joblib.load("models/trained_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    print("[INFO] Model and scaler loaded successfully.")

    # ----- Step 2: convert dict to DataFrame (1 sample) -----
    X_input = pd.DataFrame([feature_dict])
    print(f"[DEBUG] Converted to DataFrame:\n{X_input}")

    # ----- Step 3: apply scaler -----
    X_scaled = scaler.transform(X_input)
    print(f"[DEBUG] Scaled feature vector: {X_scaled}")

    # ----- Step 4: predict probability & label -----
    prob = model.predict_proba(X_scaled)[0][1]  # probability of HIGH risk
    print(f"[DEBUG] Raw model probability (High Risk): {prob:.4f}")
    label = "High Risk" if prob >= 0.66 else "Medium Risk" if prob >= 0.33 else "Low Risk"

    # ----- Step 5: return formatted result -----
    print(f"[SUMMARY] Inference completed. Assigned risk category: {label}")
    return {
        "risk_category": label,
        "risk_probability": round(float(prob), 4),
    }


def evaluate_model():
    """
    Unified evaluation covering RQ1–RQ3:
    RQ1: Assess synthetic data realism.
    RQ2: Evaluate model predictive performance.
    RQ3: Summarize key metrics for interpretability.
    """
    print("[INFO] Starting unified model evaluation (RQ1–RQ3).")

    # Fetch dataset
    df = fetch_student_data()
    print(f"[INFO] Dataset loaded: {len(df)} records, {len(df.columns)} features.")

    # RQ1: Synthetic data realism
    desc = df.describe().T
    corr = df.corr(numeric_only=True)
    print("[DEBUG] Synthetic data statistics and correlation matrix generated.")

    # Prepare training data
    X, y = prepare_data(df)
    print(f"[INFO] Prepared data: {X.shape[0]} samples, {X.shape[1]} features.")

    # Load trained model and scaler
    model = joblib.load("models/trained_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    print("[INFO] Model and scaler loaded successfully.")

    # Scale data and split
    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model evaluation
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_test, y_score),
    }
    print("[INFO] Model performance evaluated successfully.")

    # Classification report and confusion matrix
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("[INFO] Generated classification report and confusion matrix.")
    print(report)

    # Output for dashboard (RQ3 handled in Streamlit)
    print("[SUMMARY] Evaluation completed. Metrics ready for visualization.")
    return {
        "desc": desc,                   # RQ1: statistics
        "corr": corr,                   # RQ1: correlation
        "metrics": metrics,             # RQ2: key scores
        "report": report,               # RQ2: text report
        "confusion_matrix": cm,         # RQ2: visualization
    }
