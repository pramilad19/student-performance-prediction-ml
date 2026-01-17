from flask import Flask, request, jsonify
import pickle
import numpy as np
import sqlite3

app = Flask(__name__)

def get_connection():
    return sqlite3.connect("students.db")

def create_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attendance INTEGER,
            internal_marks INTEGER,
            assignment_score INTEGER,
            study_hours INTEGER,
            performance TEXT
        )
    """)
    conn.commit()
    conn.close()

create_table()

model = pickle.load(open("student_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([[
        data["attendance"],
        data["internal_marks"],
        data["assignment_score"],
        data["study_hours"]
    ]])

    prediction = model.predict(features)
    result = encoder.inverse_transform(prediction)[0]

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (attendance, internal_marks, assignment_score, study_hours, performance) VALUES (?, ?, ?, ?, ?)",
        (
            data["attendance"],
            data["internal_marks"],
            data["assignment_score"],
            data["study_hours"],
            result
        )
    )
    conn.commit()
    conn.close()

    return jsonify({"predicted_performance": result})

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(port=5000)
