import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title of the Streamlit app
st.title("Linear Regression: Total Working Years vs. Monthly Income")

# Load data from the CSV file
csv_file = "HR_Dataset_Group4.5.csv"  # File name in the repository
try:
    df = pd.read_csv(csv_file, sep=";")  # Semicolon as the delimiter
except FileNotFoundError:
    st.error("CSV file not found. Please check the file path.")
    st.stop()

# Check if required columns exist
if "TotalWorkingYears" in df.columns and "MonthlyIncome" in df.columns:
    # Filter relevant columns and drop NaN values
    df_filtered = df[["TotalWorkingYears", "MonthlyIncome"]].dropna()

    # Define features (X) and target variable (y)
    X = df_filtered[["TotalWorkingYears"]]
    y = df_filtered["MonthlyIncome"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions for test data
    y_pred = model.predict(X_test)

    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display results
    st.subheader("Model Evaluation")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    # Create visualization
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', s=20, label='Data Points')  # Scatter plot with smaller circles
    ax.plot(X, model.predict(X), color='red', label='Linear Regression')  # Regression line
    ax.set_xlabel("Total Working Years")
    ax.set_ylabel("Monthly Income")
    ax.set_title("Linear Regression: Total Working Years vs. Monthly Income")
    ax.legend()

    # Display plot in Streamlit
    st.pyplot(fig)
else:
    st.error("The required columns 'TotalWorkingYears' and 'MonthlyIncome' are not found in the dataset.")

