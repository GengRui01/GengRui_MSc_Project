import pandas as pd
import numpy as np
import datetime
from sqlalchemy import create_engine, text

# --- Connect to MySQL ---
engine = create_engine("mysql+pymysql://root:123456@localhost:3306/gengrui_msc")

# --- Generate realistic student behavior data ---
def generate_student_behavior_data(num_students):
    np.random.seed(42)
    student_ids = [f"S{str(i).zfill(3)}" for i in range(1, num_students + 1)]
    # Simulate features
    login_count = np.random.poisson(lam=10, size=num_students)
    time_spent = np.round(np.random.normal(loc=6, scale=2, size=num_students), 1)
    quiz_attempts = np.random.randint(1, 6, num_students)
    # Ensure non-negative values
    time_spent = np.clip(time_spent, 1, None)
    # Simulate completion rate based on engagement level
    completion_rate = np.round(np.clip(0.4 + 0.05 * (time_spent / time_spent.max()) + np.random.normal(0, 0.1, num_students), 0, 1), 2,)
    # Simulate quiz scores with mild correlation to completion rate
    quiz_score = np.round(np.clip(50 + 40 * completion_rate + np.random.normal(0, 10, num_students), 0, 100), 1,)
    # Simulate course progress (e.g., % of modules completed)
    progress = np.round(completion_rate * 100, 1)
    created_at = [
        (datetime.datetime.now() - datetime.timedelta(days=np.random.randint(0, 30))).strftime("%Y-%m-%d %H:%M:%S")
        for _ in range(num_students)
    ]

    df = pd.DataFrame(
        {
            "student_id": student_ids,
            "login_count": login_count,
            "time_spent": time_spent,
            "quiz_attempts": quiz_attempts,
            "completion_rate": completion_rate,
            "quiz_score": quiz_score,
            "progress": progress,
            "created_at": created_at,
        }
    )
    return df


# --- Insert data ---
def insert_data():
    # Generate 200 sample data
    df = generate_student_behavior_data(200)
    df.to_sql("student_behavior", con=engine, if_exists="append", index=False)
    print(f"[INFO] Inserted {len(df)} new student records into 'student_behavior' table.")


if __name__ == "__main__":
    insert_data()