import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score

# Titel der Streamlit-App
st.title("Linear Regression: Total Working Years vs. Monthly Income")

# Daten aus der CSV-Datei laden
csv_file = "HR_Dataset_Group4.5.csv"  # Dateiname im Repository
try:
    df = pd.read_csv(csv_file, sep=";")  # Semikolon als Trennzeichen
except FileNotFoundError:
    st.error("CSV file not found. Please check the file path.")
    st.stop()

# Prüfen, ob die erforderlichen Spalten existieren
if "TotalWorkingYears" in df.columns and "MonthlyIncome" in df.columns:
    # Relevante Spalten filtern und NaN-Werte entfernen
    df_filtered = df[["TotalWorkingYears", "MonthlyIncome"]].dropna()

    # Features und Zielvariable definieren
    X = df_filtered[["TotalWorkingYears"]].values  # Nur TotalWorkingYears als Input
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
    total_working_years = st.number_input("Enter Total Working Years", min_value=0, max_value=50, step=1)

    predicted_income = None  # Initialisierung

    if st.button("Predict Income"):
        # Neue Eingaben standardisieren
        input_data = np.array([[total_working_years]])
        input_scaled = scaler.transform(input_data)

        # Vorhersage basierend auf Eingaben
        predicted_income = model.predict(input_scaled)[0]
        st.write(f"Predicted Monthly Income: ${predicted_income:.2f}")

    # Konfidenzintervalle berechnen
    preds = []
    for _ in range(100):  # Bootstrap-Resampling
        X_sample, y_sample = resample(X_train, y_train)
        model.fit(X_sample, y_sample)
        preds.append(model.predict(X_scaled))
    lower = np.percentile(preds, 2.5, axis=0)
    upper = np.percentile(preds, 97.5, axis=0)

    # Hauptplot mit Konfidenzintervallen und Datenpunkten
    st.subheader("Visualization: Regression with Confidence Intervals")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        X.flatten(), y, color="blue", s=10, alpha=0.5, label="Data Points"
    )

    # Konfidenzintervall hinzufügen
    ax.fill_between(
        X.flatten(), lower, upper, color="gray", alpha=0.3, label="Confidence Interval"
    )

    # Regressionslinie
    ax.plot(X.flatten(), model.predict(X_scaled), color="red", linewidth=2, label="Regression Line")

    # Wenn Vorhersage gemacht wurde, füge schwarzen Punkt hinzu
    if predicted_income is not None:
        ax.scatter(total_working_years, predicted_income, color="black", s=100, label="Predicted Income", zorder=5)

    # Achsentitel und Beschriftungen
    ax.set_xlabel("Total Working Years (Years)", fontsize=12)
    ax.set_ylabel("Monthly Income (in USD)", fontsize=12)
    ax.set_title("Linear Regression: Total Working Years vs. Monthly Income", fontsize=14)
    ax.legend()
    st.pyplot(fig)

else:
    st.error("The required columns 'TotalWorkingYears' and 'MonthlyIncome' are not found in the dataset.")


