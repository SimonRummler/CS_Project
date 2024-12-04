import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Header
st.title("Multiple Regression: Total Working Years and Job Level vs. Predicted Monthly Income")

#Path and Error if access is not possible
csv_file = "HR_Dataset_Group4.5.csv"  # Lokale Datei im gleichen Verzeichnis
try:
    # CSV-Datei einlesen
    df = pd.read_csv(csv_file, sep=";")  # Semikolon als Trennzeichen
except FileNotFoundError:
    st.error("CSV-Datei nicht gefunden. Stelle sicher, dass die Datei korrekt benannt und im richtigen Verzeichnis ist.")
    st.stop()

#Sorting only the necessary data and error if not found
try:
    # Selektiere nur die relevanten Spalten für die Regression
    df = df[['TotalWorkingYears', 'JobLevel', 'MonthlyIncome']].dropna()
except KeyError as e:
    st.error(f"Fehlende Spalten: {e}")
    st.stop()

# X and y data for and training them 
X = df[['TotalWorkingYears', 'JobLevel']]
y = df['MonthlyIncome']

# Datenaufteilung in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Regressiontraining through sklearn formula
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction with method
y_pred = model.predict(X_test)


# Userinterface: using input of working years (max 40) and Job Level(1-5) with streamlit
st.subheader("Predict Monthly Income")
user_working_years = st.number_input("Enter Total Working Years:", min_value=0, max_value=40, step=1, value=10)
user_job_level = st.number_input("Enter Job Level:", min_value=1, max_value=5, step=1, value=2)

# Prediction with user input data while making data frame with entered variables 
user_input = pd.DataFrame({'TotalWorkingYears': [user_working_years], 'JobLevel': [user_job_level]})
predicted_income = model.predict(user_input)[0]
st.write(f"Predicted Monthly Income: **{predicted_income:.2f}**")

# Visualization: Scatterplot with farbcoded joblevel for better overview, the x-axis displays total working years and the y-axis shows predicted income, 
# with point colors assigned via the viridis colormap for clear differentiation of job levels + color bar is added to show the job levels
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(X_test['TotalWorkingYears'], y_pred, c=X_test['JobLevel'], cmap='viridis', s=50, alpha=0.8)
ax.set_xlabel("Total Working Years")
ax.set_ylabel("Predicted Monthly Income")
ax.set_title("Predicted Monthly Income vs. Total Working Years (Color: Job Level)")
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Job Level")

# Point of user output on graph and showing graph
ax.scatter(user_working_years, predicted_income, color='red', s=100, label='Your Input', zorder=5)
ax.legend()
st.pyplot(fig)

