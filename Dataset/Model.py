import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("E:/PCCOE/Semesters/6th/ML/Mini Project/Dataset/student_data.csv")

# Define features and labels
X = df[['study_hours', 'attendance', 'assignment_scores', 'last_sem_percentage', 'mobile_screen_time', 'sleep_hours']]
y_regression = df['final_percentage']  # Target for regression
y_classification = df['pass_fail']  # Target for classification (1 = Pass, 0 = Fail)

# Train Regression Model
X_train, X_test, y_train, y_test = train_test_split(X, y_regression, test_size=0.2, random_state=42)
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# Train Classification Model
X_train, X_test, y_train, y_test = train_test_split(X, y_classification, test_size=0.2, random_state=42)
classification_model = RandomForestClassifier(n_estimators=100, random_state=42)
classification_model.fit(X_train, y_train)

# Save the models
pickle.dump(regression_model, open("E:/PCCOE/Semesters/6th/ML/Mini Project/Dataset/regression_model.pkl", "wb"))
pickle.dump(classification_model, open("E:/PCCOE/Semesters/6th/ML/Mini Project/Dataset/classification_model.pkl", "wb"))

print("Models trained and saved successfully!")
