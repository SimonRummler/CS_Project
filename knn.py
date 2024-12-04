import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Title of the Streamlit app
st.title("Employee Attrition Prediction")

# Load data from the CSV file
csv_file = "ADJ_HR_Data.csv"  # Local file in the same directory
try:
    df = pd.read_csv(csv_file, sep=";")  # Semicolon as delimiter
except FileNotFoundError:
    st.error("CSV file not found. Please ensure the file is named correctly and located in the right directory.")
    st.stop()

# Data validation and feature scaling
try:
    # Select relevant columns
    df = df[['JobSatisfaction', 'PercentSalaryHike', 'MonthlyIncome', 'Attrition']].dropna()

    # Convert Attrition to binary (Yes=1, No=0)
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Scale numerical features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[['JobSatisfaction', 'PercentSalaryHike', 'MonthlyIncome']])
    df[['JobSatisfaction', 'PercentSalaryHike', 'MonthlyIncome']] = scaled_features
except KeyError as e:
    st.error(f"Missing columns: {e}")
    st.stop()

# Split data into features and target variable
X = df[['JobSatisfaction', 'PercentSalaryHike', 'MonthlyIncome']]
y = df['Attrition']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train k-NN classifier
st.subheader("Training k-NN Model")
k = st.slider("Select the number of neighbors (k):", min_value=1, max_value=20, value=5, step=1)
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)

# Make predictions
y_pred = knn_model.predict(X_test)

# Evaluation metrics
st.subheader("Model Evaluation")
conf_matrix = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix:")
st.write(conf_matrix)

report = classification_report(y_test, y_pred)
st.write("Classification Report:")
st.text(report)

# User interaction for prediction
st.subheader("Make a Prediction")
job_satisfaction = st.slider("Job Satisfaction (scaled 0-1):", min_value=0.0, max_value=1.0, step=0.01)
percent_salary_hike = st.slider("Percent Salary Hike (scaled 0-1):", min_value=0.0, max_value=1.0, step=0.01)
monthly_income = st.slider("Monthly Income (scaled 0-1):", min_value=0.0, max_value=1.0, step=0.01)

# Predict attrition for user input
user_input = pd.DataFrame({
    'JobSatisfaction': [job_satisfaction],
    'PercentSalaryHike': [percent_salary_hike],
    'MonthlyIncome': [monthly_income]
})
predicted_attrition = knn_model.predict(user_input)[0]

st.write(f"Predicted Attrition: {'Yes' if predicted_attrition == 1 else 'No'}")
