import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Titel der Streamlit-App
st.title("3D Regression Model: Predict Monthly Income based on Job Level and Working Years")

# Daten aus der CSV-Datei laden
csv_file = "HR_Dataset_Group4.5.csv"
try:
    df = pd.read_csv(csv_file, sep=";")
except FileNotFoundError:
    st.error("CSV file not found. Please check the file path.")
    st.stop()

# Prüfen, ob die erforderlichen Spalten existieren
required_columns = ["TotalWorkingYears", "MonthlyIncome", "JobLevel"]
if not all(col in df.columns for col in required_columns):
    st.error(f"The required columns {required_columns} are not found in the dataset.")
    st.stop()

# Relevante Spalten filtern und NaN-Werte entfernen
df_filtered = df[required_columns].dropna()

# Features und Zielvariable definieren
X = df_filtered[["TotalWorkingYears", "JobLevel"]].values
y = df_filtered["MonthlyIncome"].values

# Standardisierung der Eingabedaten
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modell trainieren
model = LinearRegression()
model.fit(X_train, y_train)

# Vorhersagen für Testdaten
y_pred = model.predict(X_test)

# Modellbewertung
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Ergebnisse anzeigen
st.subheader("Model Evaluation")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Eingabefelder für Benutzer
st.subheader("Predict Monthly Income")
job_level = st.number_input("Enter Job Level", min_value=1, max_value=5, step=1)
total_working_years = st.number_input("Enter Total Working Years", min_value=0, max_value=50, step=1)

if st.button("Predict"):
    input_data = np.array([[total_working_years, job_level]])
    input_scaled = scaler.transform(input_data)
    predicted_income = model.predict(input_scaled)[0]
    st.write(f"Predicted Monthly Income: ${predicted_income:.2f}")

# 3D-Visualisierung
st.subheader("3D Visualization of the Regression Model")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Originaldaten plotten
ax.scatter(
    df_filtered["TotalWorkingYears"],
    df_filtered["JobLevel"],
    df_filtered["MonthlyIncome"],
    c="blue",
    alpha=0.5,
    label="Data Points",
)

# Regressionsfläche erstellen
x_surf, y_surf = np.meshgrid(
    np.linspace(df_filtered["TotalWorkingYears"].min(), df_filtered["TotalWorkingYears"].max(), 50),
    np.linspace(df_filtered["JobLevel"].min(), df_filtered["JobLevel"].max(), 50),
)
z_surf = model.predict(scaler.transform(np.c_[x_surf.ravel(), y_surf.ravel()])).reshape(x_surf.shape)

# Regressionsfläche plotten
ax.plot_surface(x_surf, y_surf, z_surf, color="red", alpha=0.3, label="Regression Surface")

# Achsenbeschriftungen
ax.set_xlabel("Total Working Years")
ax.set_ylabel("Job Level")
ax.set_zlabel("Monthly Income")
ax.set_title("3D Regression: Total Working Years & Job Level vs. Monthly Income")

# Legende hinzufügen
ax.legend()
st.pyplot(fig)


