import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st

# Path 1 (this should be all new when using the API)
data_file = "New_HR_Dataset_Github_Ready.csv"
df = pd.read_csv(data_file, delimiter=',')  # Adjust the delimiter if needed

# 1.2 Strip whitespace from column names
df.columns = df.columns.str.strip()

# 1.3 Display column names for debugging
st.write("Available Columns:", df.columns.tolist())
# 1.4 Load data
data_file = "New_HR_Dataset_Github_Ready.csv"  # Ensure the correct file name and location
df = pd.read_csv(data_file, delimiter=';')

# Streamlit app setup
st.title("k-NN Regression: Predict PerformanceRating")

# Validate the dataset
st.write("Dataset Overview:")
st.write(df.head())

# Check for missing values in key columns
if df[['PercentSalaryHike', 'PerformanceRating', 'UID']].isnull().any().any():
    st.error("Dataset contains missing values. Please clean the data and try again.")
    st.stop()

# Select relevant features and target
X = df[['PercentSalaryHike']]  # Input feature
y = df['PerformanceRating']    # Target variable

# Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train the k-NN regressor
k = 5  # Number of neighbors
knn = KNeighborsRegressor(n_neighbors=k)
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display model evaluation
st.subheader("Model Evaluation")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# User input for Employee UID and PercentSalaryHike
st.subheader("Predict PerformanceRating for an Employee")
employee_uid = st.selectbox("Select Employee UID:", df['UID'])
selected_employee = df[df['UID'] == employee_uid]

st.write("Selected Employee Details:")
st.write(selected_employee)

percent_hike = st.number_input(
    "Enter PercentSalaryHike:",
    min_value=0,
    max_value=100,
    value=int(selected_employee['PercentSalaryHike'].values[0])
)

# Scale the input PercentSalaryHike
percent_hike_scaled = scaler.transform([[percent_hike]])

# Predict the PerformanceRating
predicted_performance = knn.predict(percent_hike_scaled)[0]
st.write(f"Predicted PerformanceRating: **{predicted_performance:.2f}**")

# Visualization with Matplotlib
st.subheader("Visualization of Predictions")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_test, y_test, color='blue', label='Actual Values', alpha=0.6)
ax.scatter(X_test, y_pred, color='red', label='Predicted Values', alpha=0.6)
ax.set_title(f"k-NN Regression (k={k})")
ax.set_xlabel('PercentSalaryHike (Scaled)')
ax.set_ylabel('PerformanceRating')
ax.legend()

# Display the plot
st.pyplot(fig)

st.pyplot(fig)
