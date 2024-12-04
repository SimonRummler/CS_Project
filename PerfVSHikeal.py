import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Title of the app
st.title("Predict PerformanceRating Using JobSatisfaction and PercentSalaryHike (Multiple Regression)")

# Load the dataset
csv_file = "Adjusted_HR_Dataset_Group4.5.csv"  # Replace with your file name
try:
    df = pd.read_csv(csv_file, sep=";")
except FileNotFoundError:
    st.error("The dataset file was not found. Please upload it.")
    st.stop()

# Validate required columns
required_columns = ['EmployeeNumber', 'JobSatisfaction', 'PercentSalaryHike', 'PerformanceRating']
if not all(column in df.columns for column in required_columns):
    st.error("The dataset must contain the columns: EmployeeNumber, JobSatisfaction, PercentSalaryHike, PerformanceRating.")
    st.stop()

# Define features and target variable
X = df[['JobSatisfaction', 'PercentSalaryHike']]
y = df['PerformanceRating']

# Split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a multiple linear regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Display model evaluation
st.write(f"### Model Evaluation")
st.write(f"**R² Score:** {r2:.4f}")
st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")

# Input for EmployeeNumber
st.write("### Predict PerformanceRating for a Specific Employee")
employee_number = st.number_input(
    "Enter EmployeeNumber:",
    min_value=int(df['EmployeeNumber'].min()),
    max_value=int(df['EmployeeNumber'].max()),
    step=1
)

if employee_number in df['EmployeeNumber'].values:
    # Get current JobSatisfaction and PercentSalaryHike for the employee
    employee_data = df.loc[df['EmployeeNumber'] == employee_number]
    current_job_satisfaction = employee_data['JobSatisfaction'].values[0]
    current_percent_hike = employee_data['PercentSalaryHike'].values[0]

    st.write(f"Current JobSatisfaction: {current_job_satisfaction}")
    st.write(f"Current PercentSalaryHike: {current_percent_hike}%")

    # Inputs to adjust JobSatisfaction and PercentSalaryHike
    new_job_satisfaction = st.slider(
        "Adjust JobSatisfaction:",
        min_value=1, max_value=5,
        value=int(current_job_satisfaction), step=1
    )

    new_percent_hike = st.slider(
        "Adjust PercentSalaryHike:",
        min_value=0, max_value=30,
        value=int(current_percent_hike), step=1
    )

    # Predict PerformanceRating based on the adjusted inputs
    prediction = reg.predict([[new_job_satisfaction, new_percent_hike]])[0]
    prediction = np.clip(np.round(prediction), 1, 5)  # Round to nearest integer and clip to range [1, 5]
    st.write(f"Predicted PerformanceRating for EmployeeNumber {employee_number}: {int(prediction)}")
else:
    st.error("The entered EmployeeNumber is not found in the dataset.")

