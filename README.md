# 🎓 Student Behavior Analysis and Prediction Dashboard

This project implements a machine learning-based dashboard for analysing and predicting student learning risks on online platforms.  
It was developed as part of the MSc Project — *Student Behavior Analysis and Prediction on Online Learning Platforms Based on Machine Learning*.

---

## 📂 Project Structure

```
GengRui_MSc_Project/
│
├── app.py                     # Streamlit main dashboard
│
├── db/
│   └── init.sql               # MySQL database initialization script
│
├── models/
│   ├── trained_model.pkl      # Trained ML model
│   └── scaler.pkl             # StandardScaler object
│
├── utils/
│   ├── data_generator.py      # Generate or load student behavior data
│   ├── db_connector.py        # Connect to MySQL database
│   └── model_trainer.py       # Model training and evaluation logic
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🧩 Environment Setup

To set up the environment, install all required Python packages with a single command:  

```bash
pip install streamlit pandas numpy scikit-learn sqlalchemy pymysql plotly
```

After successful installation, you can verify the environment with the following commands:  

```bash
streamlit --version
python -m sklearn --help
```

---

# 🚀 How to Run

### 1. Initialize Database
Run the SQL script once to create the database and tables:
```bash
mysql -u root -p < db/init.sql
```

### 2. Launch Dashboard
Start the Streamlit application:
```bash
streamlit run app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501).

> 💡 Note:
> - The system will automatically connect to the database and load the pre-trained model.
> - The pre-trained model (`trained_model.pkl`) and scaler (`scaler.pkl`) are included to ensure result reproducibility.
> - If you wish to retrain the model, you can manually run:
>   ```bash
>   python utils/model_trainer.py
>   ```
>   This will update the model files under the `models/` directory.

---

## 📚 Author
**Geng Rui (耿锐)**  
MSc Project – *Student Behavior Analysis and Prediction on Online Learning Platforms Based on Machine Learning*  
Supervisor: *(fill in your supervisor name)*  
