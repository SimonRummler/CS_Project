train_test_split(X, y, test_size=0.2, random_state=42)

# Train regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Evaluate model
y_pred = reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
st.write("### Model Evaluation")
st.write(f"**R² Score:** {r2:.4f}")
st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")

# Employee-specific predictions
st.write("### Predict for Specific Employee")
employee_id = st.number_input(
    "Enter Employee Number:",
    min_value=int(data['EmployeeNumber'].min()),
    max_value=int(data['EmployeeNumber'].max()),
    step=1
)

if employee_id in data['EmployeeNumber'].values:
    # Extract employee data
    employee_data = data[data['EmployeeNumber'] == employee_id].iloc[0]
    current_job_satisfaction = employee_data['JobSatisfaction']
    current_percent_hike = employee_data['PercentSalaryHike']
    current_performance_rating = employee_data['PerformanceRating']
    monthly_income = employee_data['MonthlyIncome']

    # Calculate current salary increase
    current_salary_increase = (current_percent_hike / 100) * monthly_income

    st.write(f"**Current JobSatisfaction:** {current_job_satisfaction}")
    st.write(f"**Current PercentSalaryHike:** {current_percent_hike:.2f}%")
    st.write(f"**Current PerformanceRating:** {current_performance_rating}")
    st.write(f"**Current Salary Increase:** ${current_salary_increase:,.2f}")

    # Inputs to adjust JobSatisfaction and PercentSalaryHike
    new_job_satisfaction = st.slider(
        "Adjust JobSatisfaction:",
        min_value=1,
        max_value=5,
        value=int(current_job_satisfaction),
        step=1
    )

    new_percent_hike = st.slider(
        "Adjust PercentSalaryHike:",
        min_value=0,
        max_value=30,
        value=int(current_percent_hike),
        step=1
    )

    # Calculate new salary increase
    new_salary_increase = (new_percent_hike / 100) * monthly_income

    # Predict PerformanceRating based on the adjusted inputs
    prediction = reg.predict([[new_job_satisfaction, new_percent_hike]])[0]
    prediction = np.clip(np.round(prediction), 1, 5)

    st.write(f"**Predicted PerformanceRating for EmployeeNumber {employee_id}: {int(prediction)}**")
    st.write(f"**New Salary Increase:** ${new_salary_increase:,.2f}")
else:
    st.error("The entered Employee Number is not found.")
