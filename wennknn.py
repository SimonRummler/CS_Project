import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Header
st.title("KNN Regression: Predict Monthly Income using Total Working Years and Job Level")

# Beispiel-Datenquelle (aus Notebooks extrahiert)
# Dieser Teil kann angepasst werden, falls eine andere Datenquelle benötigt wird.
csv_url = 'https://drive.switch.ch/index.php/s/wWTGBFrSSCTCphU/download'  # Beispiel-URL
try:
    df = pd.read_csv(csv_url)  # Daten laden
    st.success("Daten erfolgreich geladen!")
except Exception as e:
    st.error(f"Fehler beim Laden der Daten: {e}")
    st.stop()

# Spaltenauswahl (anpassbar an den genauen Datensatz)
try:
    df = df[['TotalWorkingYears', 'JobLevel', 'MonthlyIncome']].dropna()
except KeyError as e:
    st.error(f"Fehlende Spalten: {e}")
    st.stop()

# Datenaufteilung
X = df[['TotalWorkingYears', 'JobLevel']]
y = df['MonthlyIncome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Benutzeroberfläche für Parameter
st.subheader("Parameter für KNN-Regressor")
k = st.slider("Wähle Anzahl der Nachbarn (k):", min_value=1, max_value=20, value=5)

# Pipeline mit Skalierung und KNN-Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(n_neighbors=k))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Benutzereingabe für Vorhersage
user_working_years = st.number_input("Total Working Years:", min_value=0, max_value=40, step=1, value=10)
user_job_level = st.number_input("Job Level:", min_value=1, max_value=5, step=1, value=2)

# Vorhersage
user_input = pd.DataFrame({'TotalWorkingYears': [user_working_years], 'JobLevel': [user_job_level]})
predicted_income = pipeline.predict(user_input)[0]
st.write(f"Vorhergesagtes Einkommen: **{predicted_income:.2f}**")

# Visualisierung
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(X_test['TotalWorkingYears'], y_pred, c=X_test['JobLevel'], cmap='viridis', s=50, alpha=0.8)
ax.set_xlabel("Total Working Years")
ax.set_ylabel("Predicted Monthly Income")
ax.set_title("Predicted Monthly Income vs. Total Working Years")
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Job Level")
ax.scatter(user_working_years, predicted_income, color='red', s=100, label='Your Input', zorder=5)
ax.legend()
st.pyplot(fig)
