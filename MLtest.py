import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Titel der Streamlit-App
st.title("Lineare Regression: Performance vs. Gehalt")

# Beispiel-Daten erstellen
data = {
    "Performance": [3.8, 4.2, 4.5, 4.0, 3.6, 4.8, 3.9],
    "Salary": [50000, 55000, 60000, 52000, 48000, 65000, 51000]
}
df = pd.DataFrame(data)

# Anzeige der Daten in Streamlit
st.subheader("Daten")
st.dataframe(df)

# Features und Zielvariable definieren
X = df[["Performance"]]
y = df["Salary"]

# Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell trainieren
model = LinearRegression()
model.fit(X_train, y_train)

# Vorhersagen für Testdaten
y_pred = model.predict(X_test)

# Modellbewertung
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Ergebnisse anzeigen
st.subheader("Modellbewertung")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Visualisierung erstellen
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='Datenpunkte')  # Scatterplot
ax.plot(X, model.predict(X), color='red', label='Lineare Regression')  # Regressionslinie
ax.set_xlabel("Performance-Bewertung")
ax.set_ylabel("Gehalt (in €)")
ax.set_title("Lineare Regression")
ax.legend()

# Plot in Streamlit anzeigen
st.pyplot(fig)
