import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

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

# Debug: Vorschau der Daten
st.write("DataFrame Preview:")
st.write(df_filtered.head())

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

# Vorhersagen für alle Datenpunkte
y_pred_full = model.predict(X_scaled)

# Sicherstellen, dass y_pred_full definiert ist
if "y_pred_full" not in locals() or len(y_pred_full) == 0:
    st.error("The predictions (y_pred_full) are not defined or empty. Check your model training.")
    st.stop()

# Modellbewertung
mse = mean_squared_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))

# Berechnung zusätzlicher Metriken
n = len(y)  # Anzahl der Beobachtungen
k = X.shape[1]  # Anzahl der Prädiktoren
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))  # Adjusted R²
rmse = np.sqrt(mse)  # Root Mean Squared Error

# Modellbewertung anzeigen
st.subheader("Model Metrics")
st.write(f"R²: {r2:.2f}")
st.write(f"Adjusted R²: {adjusted_r2:.2f}")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# **3D-Visualisierung mit 30-Grad-Drehung**
st.subheader("3D Visualization with 30-Degree Rotation to the Left")
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Datenpunkte mit Farbcodierung nach JobLevel
job_levels = df_filtered["JobLevel"].unique()
colors = sns.color_palette("viridis", len(job_levels))

for i, job_level in enumerate(sorted(job_levels)):
    # Filter für das jeweilige Job Level
    level_data = df_filtered[df_filtered["JobLevel"] == job_level]
    if level_data.empty:
        st.warning(f"No data found for Job Level {job_level}.")
        continue

    level_predictions = y_pred_full[df_filtered["JobLevel"] == job_level]

    # Punkte über der Regressionslinie markieren
    above_line = level_data["MonthlyIncome"] > level_predictions
    below_line = ~above_line

    # Punkte über der Linie
    ax.scatter(
        level_data["TotalWorkingYears"][above_line],
        level_data["JobLevel"][above_line],
        level_data["MonthlyIncome"][above_line],
        color=colors[i],
        alpha=0.8,
        label=f"Job Level {job_level} (Above Line)",
        marker="^",
    )

    # Punkte unter der Linie
    ax.scatter(
        level_data["TotalWorkingYears"][below_line],
        level_data["JobLevel"][below_line],
        level_data["MonthlyIncome"][below_line],
        color=colors[i],
        alpha=0.5,
        label=f"Job Level {job_level} (Below Line)",
        marker="o",
    )

# Regressionsfläche erstellen
x_surf, y_surf = np.meshgrid(
    np.linspace(df_filtered["TotalWorkingYears"].min(), df_filtered["TotalWorkingYears"].max(), 50),
    np.linspace(df_filtered["JobLevel"].min(), df_filtered["JobLevel"].max(), 50),
)
z_surf = model.predict(scaler.transform(np.c_[x_surf.ravel(), y_surf.ravel()])).reshape(x_surf.shape)

# Regressionsfläche plotten
ax.plot_surface(x_surf, y_surf, z_surf, cmap="viridis", alpha=0.3, edgecolor="none")

# Achsentitel und Beschriftungen
ax.set_xlabel("Total Working Years")
ax.set_ylabel("Job Level")
ax.set_zlabel("Monthly Income")
ax.set_title("3D Regression with 30-Degree Rotation to the Left")

# **Anpassung der Perspektive**
ax.view_init(elev=10, azim=-30)  # Perspektive: Elevation bleibt 10, Azimut um 30 Grad nach links

# Legende hinzufügen
ax.legend(loc="best")
st.pyplot(fig)
