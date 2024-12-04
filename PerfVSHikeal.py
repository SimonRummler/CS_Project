import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# App title
st.title("Predict PerformanceRating Using JobSatisfaction and PercentSalaryHike")

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path, sep=";")

try:
    data = load_data("Adjusted_HR_Dataset_Group4.5.csv")
except FileNotFoundError:
    st.error("Dataset file not found. Please ensure the file exists in the working directory.")
    st.stop()

# Validate dataset
required_columns = ['EmployeeNumber', 'JobSatisfaction', 'PercentSalaryHike', 'PerformanceRating', 'MonthlyIncome']
if not all(column in data.columns for column in required_columns):
    st.error("Dataset must contain the required columns.")
    st.stop()

# Split data
X = data[['JobSatisfaction', 'PercentSalaryHike']]
y = data['PerformanceRating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Evaluate model
y_pred = reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
st.write("### Model Evaluation")
st.write(f"R²: {r2:.4f}")
st.write(f"MSE: {mse:.4f}")

# Employee-specific predictions
st.write("### Predict for Specific Employee")
employee_id = st.number_input("Enter Employee Number:", min_value=int(data['EmployeeNumber'].min()), max_value=int(data['EmployeeNumber'].max()), step=1)

if employee_id in data['EmployeeNumber'].values:
    employee_data = data[data['EmployeeNumber'] == employee_id].iloc[0]
    st.write(f"Current JobSatisfaction: {employee_data['JobSatisfaction']}")
    st.write(f"Current PercentSalaryHike: {employee_data['PercentSalaryHike']:.2f}%")
    st.write(f"Current PerformanceRating: {employee_data['PerformanceRating']}")

    # Adjust inputs
    job_satisfaction = st.slider("Adjust JobSatisfaction:", 1, 5, int(employee_data['JobSatisfaction']), step=1)
    salary_hike = st.slider("Adjust PercentSalaryHike:", 0, 30, int(employee_data['PercentSalaryHike']), step=1)

    # Predict new performance rating
    predicted_rating = reg.predict([[job_satisfaction, salary_hike]])[0]
    predicted_rating = np.clip(round(predicted_rating), 1, 5)

    # Calculate salary increase
    salary_increase = (salary_hike / 100) * employee_data['MonthlyIncome']

    st.write(f"Predicted PerformanceRating: {predicted_rating}")
    st.write(f"Salary Increase: ${salary_increase:,.2f}")
else:
    st.error("Employee number not found.")


