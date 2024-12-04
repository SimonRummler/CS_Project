import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st

# Title of the Streamlit App
st.title("Linear Regression: PercentSalaryHike vs. PerformanceRating")

# Load data from the CSV file
csv_file = "HR_Dataset_Group4.5.csv"  # Local file in the same directory
try:
    # Read the CSV file
    df = pd.read_csv(csv_file, sep=";")  # Semicolon delimiter
except FileNotFoundError:
    st.error("CSV file not found. Please ensure the file is correctly named and in the right directory.")
    st.stop()

# Validate data and select relevant columns
try:
    df = df[['PercentSalaryHike', 'PerformanceRating']].dropna()  # Only relevant columns
except KeyError as e:
    st.error(f"Missing columns: {e}")
    st.stop()

# Define features and target
X = df[['PerformanceRating']]
y = df['PercentSalaryHike']

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
st.write(f"Model Coefficient: {model.coef_[0]:.2f}")
st.write(f"Intercept: {model.intercept_:.2f}")

# User-friendly prediction
st.subheader("Prediction for User Inputs")
user_performance_rating = st.number_input("Enter PerformanceRating:", min_value=1, max_value=4, step=1, value=3)

# Prediction for user input
user_input = pd.DataFrame({'PerformanceRating': [user_performance_rating]})
predicted_hike = model.predict(user_input)[0]
st.write(f"Predicted PercentSalaryHike: **{predicted_hike:.2f}**")

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X_test, y_test, color='blue', label='Actual Values', alpha=0.6)
ax.plot(X_test, y_pred, color='red', label='Regression Line', linewidth=2)
ax.set_xlabel("PerformanceRating")
ax.set_ylabel("PercentSalaryHike")
ax.set_title("PercentSalaryHike vs. PerformanceRating with Regression Line")
ax.legend()

# Add user input point
ax.scatter(user_performance_rating, predicted_hike, color='green', s=100, label='User Input', zorder=5)
ax.legend()

# Display plot
st.pyplot(fig)
