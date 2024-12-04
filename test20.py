import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title of the Streamlit app
st.title("Multiple Regression: Job Satisfaction and Percent Salary Hike vs. Performance Rating")

# Load data from the CSV file
csv_file = "ADJ_HR_Data.csv"  # Local file in the same directory
try:
    # Read the CSV file
    df = pd.read_csv(csv_file, sep=";")  # Semicolon as delimiter
except FileNotFoundError:
    st.error("CSV file not found. Make sure the file is named correctly and in the right directory.")
    st.stop()

# Validate data
try:
    # Select only relevant columns for regression
    df = df[['UID', 'JobSatisfaction', 'PercentSalaryHike', 'PerformanceRating', 'MonthlyIncome']].dropna()
except KeyError as e:
    st.error(f"Missing columns: {e}")
    st.stop()

# Define features and target variable
X = df[['JobSatisfaction', 'PercentSalaryHike']]
y = df['PerformanceRating']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
st.subheader("Regression Results")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")
st.write("Model Coefficients:")
st.write(dict(zip(['JobSatisfaction', 'PercentSalaryHike'], model.coef_)))
st.write(f"Intercept: {model.intercept_:.2f}")

# User interaction: Select an employee
st.subheader("Select an Employee and Predict Performance Rating")
uids = df['UID'].unique()
selected_uid = st.selectbox("Select Employee UID:", uids)

# Display current values for the selected employee
employee_data = df[df['UID'] == selected_uid].iloc[0]
st.write("**Current Employee Values:**")
st.write(f"- Job Satisfaction: {employee_data['JobSatisfaction']}")
st.write(f"- Percent Salary Hike: {employee_data['PercentSalaryHike']}%")
st.write(f"- Monthly Income: {employee_data['MonthlyIncome']}")

# User input for Percent Salary Hike and Job Satisfaction
user_percent_salary_hike = st.slider(
    "Select Percent Salary Hike (%):",
    min_value=1,
    max_value=30,
    value=int(employee_data['PercentSalaryHike']),
    step=1
)
user_job_satisfaction = st.slider(
    "Select Job Satisfaction:",
    min_value=1,
    max_value=5,
    value=int(employee_data['JobSatisfaction']),
    step=1
)

# Calculate salary change based on Percent Salary Hike
current_percent_hike = employee_data['PercentSalaryHike']
salary_increase_percent = user_percent_salary_hike - current_percent_hike
current_monthly_income = employee_data['MonthlyIncome']
salary_increase_amount = current_monthly_income * (salary_increase_percent / 100)
salary_increase_amount_int = int(round(salary_increase_amount))

st.write(f"**Salary Change:** {salary_increase_amount_int} (based on a change in Percent Salary Hike of {salary_increase_percent}%)")

# Predict Performance Rating based on user input
user_input = pd.DataFrame({
    'JobSatisfaction': [user_job_satisfaction],
    'PercentSalaryHike': [user_percent_salary_hike]
})
predicted_performance_rating = model.predict(user_input)[0]
predicted_performance_rating = np.clip(predicted_performance_rating, 1, 5)  # Limit between 1 and 5

st.write(f"**Predicted Performance Rating:** {predicted_performance_rating:.2f}")


