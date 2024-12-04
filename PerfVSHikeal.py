import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Title of the app
st.title("Performance Prediction Based on JobSatisfaction and PercentSalaryHike")

# Load dataset function
def load_dataset(file_name):
    return pd.read_csv(file_name, sep=";")

# Load data
try:
    data = load_dataset("Adjusted_HR_Dataset_Group4.5.csv")  # Datei im selben Verzeichnis
except FileNotFoundError:
    st.error("Dataset file not found. Please ensure the file is available in the current directory.")
    st.stop()

# Display dataset structure
st.write("### Dataset Columns:")
st.write(list(data.columns))

# Check required columns
required_columns = ['EmployeeNumber', 'JobSatisfaction', 'PercentSalaryHike', 'PerformanceRating', 'MonthlyIncome']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    st.error(f"Missing required columns: {', '.join(missing_columns)}")
    st.stop()

# Prepare data
X = data[['JobSatisfaction', 'PercentSalaryHike']]
y = data['PerformanceRating']

# Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.write("### Model Evaluation:")
st.write(f"**R² Score:** {r2:.4f}")
st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")

# Input section for employee-specific predictions
st.write("### Employee-Specific Prediction")
employee_id = st.number_input(
    "Enter Employee Number:",
    min_value=int(data['EmployeeNumber'].min()),
    max_value=int(data['EmployeeNumber'].max()),
    step=1
)

if employee_id in data['EmployeeNumber'].values:
    emp_data = data[data['EmployeeNumber'] == employee_id].iloc[0]
    st.write(f"**Current JobSatisfaction:** {emp_data['JobSatisfaction']}")
    st.write(f"**Current PercentSalaryHike:** {emp_data['PercentSalaryHike']}%")
    st.write(f"**Current PerformanceRating:** {emp_data['PerformanceRating']}")
    st.write(f"**MonthlyIncome:** ${emp_data['MonthlyIncome']:,.2f}")

    # Calculate current salary increase
    current_salary_increase = (emp_data['PercentSalaryHike'] / 100) * emp_data['MonthlyIncome']
    st.write(f"**Current Salary Increase:** ${current_salary_increase:,.2f}")

    # Input sliders for adjustments
    new_job_satisfaction = st.slider(
        "Adjust JobSatisfaction:",
        min_value=1, max_value=5, value=int(emp_data['JobSatisfaction']), step=1
    )

    new_percent_hike = st.slider(
        "Adjust PercentSalaryHike (%):",
        min_value=0, max_value=30, value=int(emp_data['PercentSalaryHike']), step=1
    )

    # Predict new performance rating
    prediction = model.predict([[new_job_satisfaction, new_percent_hike]])[0]
    prediction = np.clip(round(prediction), 1, 5)

    # Calculate new salary increase
    new_salary_increase = (new_percent_hike / 100) * emp_data['MonthlyIncome']
    
    st.write(f"**Predicted PerformanceRating:** {prediction}")
    st.write(f"**New Salary Increase:** ${new_salary_increase:,.2f}")
else:
    st.error("Employee number not found.")

