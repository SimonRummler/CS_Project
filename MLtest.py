import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title of the Streamlit app
st.title("Linear Regression: Performance vs. Salary")

# Create sample data
data = {
    "Performance": [3.8, 4.2, 4.5, 4.0, 3.6, 4.8, 3.9],
    "Salary": [50000, 55000, 60000, 52000, 48000, 65000, 51000]
}
df = pd.DataFrame(data)

# Display the data in Streamlit
st.subheader("Data")
st.dataframe(df)

# Define features and target variable
X = df[["Performance"]]
y = df["Salary"]

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
ax.scatter(X, y, color='blue', label='Data Points')  # Scatter plot
ax.plot(X, model.predict(X), color='red', label='Linear Regression')  # Regression line
ax.set_xlabel("Performance Rating")
ax.set_ylabel("Salary (in €)")
ax.set_title("Linear Regression")
ax.legend()

# Display plot in Streamlit
st.pyplot(fig)
