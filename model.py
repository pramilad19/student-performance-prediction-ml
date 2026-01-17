import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

print("Starting model training...")

# Load dataset
data = pd.read_csv("student_data.csv")
print("Dataset loaded")

# Encode target column
encoder = LabelEncoder()
data["performance"] = encoder.fit_transform(data["performance"])
print("Label encoding done")

# Features and target
X = data[["attendance", "internal_marks", "assignment_score", "study_hours"]]
y = data["performance"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)
print("Model training completed")

# Save model and encoder
pickle.dump(model, open("student_model.pkl", "wb"))
pickle.dump(encoder, open("label_encoder.pkl", "wb"))

print("Model trained successfully and files saved")
