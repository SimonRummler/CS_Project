import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the data from the CSV file
file_path = "ADJ_HR_Data_Performance_Rating_JobSatisfaction.csv"  # Replace with your actual file path
try:
    data = pd.read_csv(file_path, sep=";")  # Adjust delimiter if needed
except FileNotFoundError:
    print("CSV file not found. Ensure the file is in the correct directory and the path is correct.")
    exit()

# Select and preprocess relevant columns
df = data[['JobSatisfaction_scaled', 'PercentSalaryHike_scaled', 'MonthlyIncome', 'Attrition']].dropna()

# Convert Attrition to binary (Yes = 1, No = 0)
df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# Rescale Job Satisfaction to 1-5
df['JobSatisfaction'] = df['JobSatisfaction_scaled'] * (5 - 1) + 1

# Rescale Percent Salary Hike to 1-30%
df['PercentSalaryHike'] = df['PercentSalaryHike_scaled'] * (30 - 1) + 1

# Rescale Monthly Income to 2000-27000
df['MonthlyIncome'] = df['MonthlyIncome'] * (27000 - 2000) + 2000

# Define features and target
X = df[['JobSatisfaction', 'PercentSalaryHike', 'MonthlyIncome']]
y = df['Attrition']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the k-NN classifier
k = 5  # You can adjust this value as needed
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)

# Make predictions
y_pred = knn_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the evaluation results
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)

# Example user input for prediction
print("\n--- User Prediction Example ---")
job_satisfaction = float(input("Enter Job Satisfaction (1-5): "))
percent_salary_hike = float(input("Enter Percent Salary Hike (1-30%): "))
monthly_income = float(input("Enter Monthly Income (2000-27000): "))

# Predict based on user input
user_input = pd.DataFrame({
    'JobSatisfaction': [job_satisfaction],
    'PercentSalaryHike': [percent_salary_hike],
    'MonthlyIncome': [monthly_income]
})
predicted_attrition = knn_model.predict(user_input)[0]
print(f"Predicted Attrition: {'Yes' if predicted_attrition == 1 else 'No'}")

