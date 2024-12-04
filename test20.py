import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Titel der Streamlit-App
st.title("Multiple Regression: Job Satisfaction und Percent Salary Hike vs. Performance Rating")

# Daten aus der CSV-Datei laden
csv_file = ""  # Lokale Datei im gleichen Verzeichnis
try:
    # CSV-Datei einlesen
    df = pd.read_csv(csv_file, sep=";")  # Semikolon als Trennzeichen
except FileNotFoundError:
    st.error("CSV-Datei nicht gefunden. Stelle sicher, dass die Datei korrekt benannt und im richtigen Verzeichnis ist.")
    st.stop()

# Daten validieren
try:
    # Selektiere nur die relevanten Spalten für die Regression
    df = df[['UID', 'JobSatisfaction', 'PercentSalaryHike', 'PerformanceRating', 'MonthlyIncome']].dropna()
except KeyError as e:
    st.error(f"Fehlende Spalten: {e}")
    st.stop()

# Features und Ziel definieren
X = df[['JobSatisfaction', 'PercentSalaryHike']]
y = df['PerformanceRating']

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
st.subheader("Regressionsergebnisse")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")
st.write("Modelkoeffizienten:")
st.write(dict(zip(['JobSatisfaction', 'PercentSalaryHike'], model.coef_)))
st.write(f"Intercept: {model.intercept_:.2f}")

# Benutzerinteraktion: Mitarbeiter auswählen
st.subheader("Mitarbeiter auswählen und Performance Rating vorhersagen")
uids = df['UID'].unique()
selected_uid = st.selectbox("Wähle die Mitarbeiter-UID:", uids)

# Aktuelle Werte des Mitarbeiters anzeigen
employee_data = df[df['UID'] == selected_uid].iloc[0]
st.write("**Aktuelle Werte des Mitarbeiters:**")
st.write(f"- Job Satisfaction: {employee_data['JobSatisfaction']}")
st.write(f"- Percent Salary Hike: {employee_data['PercentSalaryHike']}%")
st.write(f"- Monthly Income: {employee_data['MonthlyIncome']}")

# Benutzereingabe für Percent Salary Hike und Job Satisfaction
user_percent_salary_hike = st.slider(
    "Wähle Percent Salary Hike (%):",
    min_value=1,
    max_value=30,
    value=int(employee_data['PercentSalaryHike']),
    step=1
)
user_job_satisfaction = st.slider(
    "Wähle Job Satisfaction:",
    min_value=1,
    max_value=5,
    value=int(employee_data['JobSatisfaction']),
    step=1
)

# Veränderung des Percent Salary Hike berechnen und in Gehaltsänderung umrechnen
current_percent_hike = employee_data['PercentSalaryHike']
salary_increase_percent = user_percent_salary_hike - current_percent_hike
current_monthly_income = employee_data['MonthlyIncome']
salary_increase_amount = current_monthly_income * (salary_increase_percent / 100)
salary_increase_amount_int = int(round(salary_increase_amount))

st.write(f"**Veränderung des Gehalts:** {salary_increase_amount_int} (basierend auf einer Veränderung des Percent Salary Hike um {salary_increase_percent}%)")

# Vorhersage des Performance Rating für Benutzereingabe
user_input = pd.DataFrame({
    'JobSatisfaction': [user_job_satisfaction],
    'PercentSalaryHike': [user_percent_salary_hike]
})
predicted_performance_rating = model.predict(user_input)[0]
predicted_performance_rating = np.clip(predicted_performance_rating, 1, 5)  # Begrenzung zwischen 1 und 5

st.write(f"**Vorhergesagtes Performance Rating:** {predicted_performance_rating:.2f}")

