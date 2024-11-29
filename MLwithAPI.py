import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Titel der Streamlit-App
st.title("Linear Regression: Total Working Years and Job Level vs. Predicted Monthly Income")

# Daten aus der CSV-Datei laden
csv_file = "HR_Dataset_Group4.5.csv"  # Lokale Datei im gleichen Verzeichnis
try:
    # CSV-Datei einlesen
    df = pd.read_csv(csv_file, sep=";")  # Semikolon als Trennzeichen
except FileNotFoundError:
    st.error("CSV-Datei nicht gefunden. Stelle sicher, dass die Datei korrekt benannt und im richtigen Verzeichnis ist.")
    st.stop()

# Daten validieren
try:
    # Selektiere nur die relevanten Spalten für die Regression
    df = df[['TotalWorkingYears', 'JobLevel', 'MonthlyIncome']].dropna()
except KeyError as e:
    st.error(f"Fehlende Spalten: {e}")
    st.stop()

# Features und Ziel definieren
X = df[['TotalWorkingYears', 'JobLevel']]
y = df['MonthlyIncome']

# Datenaufteilung in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Regressionstraining
model = LinearRegression()
model.fit(X_train, y_train)

# Vorhersagen erstellen
y_pred = model.predict(X_test)

# Metriken berechnen
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Ergebnisse anzeigen
st.subheader("Regression Results")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")
st.write("Model Coefficients:")
st.write(dict(zip(['TotalWorkingYears', 'JobLevel'], model.coef_)))
st.write(f"Intercept: {model.intercept_:.2f}")

# Benutzereingabe für Total Working Years und Job Level
st.subheader("Predict Monthly Income")
user_working_years = st.number_input("Enter Total Working Years:", min_value=0, max_value=40, step=1, value=10)
user_job_level = st.number_input("Enter Job Level:", min_value=1, max_value=5, step=1, value=2)

# Vorhersage des Monthly Income für Benutzereingabe
user_input = pd.DataFrame({'TotalWorkingYears': [user_working_years], 'JobLevel': [user_job_level]})
predicted_income = model.predict(user_input)[0]
st.write(f"Predicted Monthly Income: **{predicted_income:.2f}**")

# Visualisierung: Scatterplot mit farbkodiertem JobLevel und Benutzereingabe
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(X_test['TotalWorkingYears'], y_pred, c=X_test['JobLevel'], cmap='viridis', s=50, alpha=0.8)
ax.set_xlabel("Total Working Years")
ax.set_ylabel("Predicted Monthly Income")
ax.set_title("Predicted Monthly Income vs. Total Working Years (Color: Job Level)")
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Job Level")

# Benutzereingabe auf dem Graphen darstellen
ax.scatter(user_working_years, predicted_income, color='black', s=100, label='Your Input', zorder=5)
ax.legend()

# Plot anzeigen
st.pyplot(fig)

