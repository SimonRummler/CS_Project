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
if "TotalWorkingYears" in df.columns and "MonthlyIncome" in df.columns and "JobLevel" in df.columns:
    # Relevante Spalten filtern und NaN-Werte entfernen
    df_filtered = df[["TotalWorkingYears", "MonthlyIncome", "JobLevel"]].dropna()

    # Features und Zielvariable definieren
    X = df_filtered[["TotalWorkingYears"]].values  # Konvertiere in numpy.ndarray
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

    # Konfidenzintervalle berechnen
    preds = []
    for _ in range(100):  # Bootstrap-Resampling
        X_sample, y_sample = resample(X_train, y_train)
        model.fit(X_sample, y_sample)
        preds.append(model.predict(X_scaled))
    lower = np.percentile(preds, 2.5, axis=0)
    upper = np.percentile(preds, 97.5, axis=0)

    # Hauptplot mit Konfidenzintervallen und Farbkodierung
    st.subheader("Visualization: Regression with Confidence Intervals")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        X.flatten(), y, c=df_filtered["JobLevel"], cmap="viridis", s=10, alpha=0.5, label="Data Points"
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Job Level")

    # Konfidenzintervall hinzufügen
    ax.fill_between(
        X.flatten(), lower, upper, color="gray", alpha=0.3, label="Confidence Interval"
    )

    # Regressionslinie
    ax.plot(X.flatten(), model.predict(X_scaled), color="red", linewidth=2, label="Regression Line")

    # Achsentitel und Beschriftungen
    ax.set_xlabel("Total Working Years (Years)", fontsize=12)
    ax.set_ylabel("Monthly Income (in USD)", fontsize=12)
    ax.set_title("Enhanced Linear Regression: Total Working Years vs. Monthly Income", fontsize=14)
    ax.legend()
    st.pyplot(fig)

   
