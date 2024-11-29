import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Titel der Streamlit-App
st.title("Linear Regression: Total Working Years and Job Level vs. Monthly Income")

# Daten aus der CSV-Datei laden
csv_file = "HR_Dataset_Group4.5.csv"  # Lokale Datei im gleichen Verzeichnis
try:
    df = pd.read_csv(csv_file, sep=";")  # Semikolon als Trennzeichen
    st.write("Daten erfolgreich geladen:")
    st.write(df.head())
except FileNotFoundError:
    st.error("CSV-Datei nicht gefunden. Bitte prüfen Sie den Dateipfad.")
    st.stop()

# Daten validieren
try:
    df = df[['TotalWorkingYears', 'JobLevel', 'MonthlyIncome']].dropna()
except KeyError as e:
    st.error(f"Fehlende Spalten: {e}")
    st.stop()

# Features und Ziel definieren
X = df[['TotalWorkingYears', 'JobLevel']]
y = df['MonthlyIncome']

# Datenaufteilung
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

# Visualisierung
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
ax.set_xlabel("Actual Monthly Income")
ax.set_ylabel("Predicted Monthly Income")
ax.set_title("Actual vs Predicted Monthly Income")
st.pyplot(fig)

