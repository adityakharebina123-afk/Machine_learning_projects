# Machine_learning_projects
import joblib
import numpy as np


model = joblib.load("../models/student_performance_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
encoder = joblib.load("../models/label_encoder.pkl")


study_hours = 5
attendance = 85
previous_score = 78
sleep_hours = 7
internet_access = 1
extracurricular = 1


student = np.array([[
    study_hours,
    attendance,
    previous_score,
    sleep_hours,
    internet_access,
    extracurricular
]])

student_scaled = scaler.transform(student)
prediction = model.predict(student_scaled)
result = encoder.inverse_transform(prediction)
print("Predicted Student Performance:", result[0])
