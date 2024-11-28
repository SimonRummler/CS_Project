import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Titel der Streamlit-App
st.title("Linear Regression: Job Satisfaction vs. Monthly Income")

# Beispiel-Daten basierend auf der Tabelle
data = {
    "Performance": [1, 1, 3, 1, 2, 3, 3, 3, 4, 4, 4, 4, 2, 3, 3, 1, 3, 1, 2, 1, 2, 4, 4, 1, 1, 4, 3, 1, 2, 1, 3, 4, 2, 4, 1, 4, 2, 1, 3, 1, 4, 1, 1, 3, 3, 4, 2, 1, 3, 2],
    "Salary": [8463, 4450, 1555, 9724, 5914, 2579, 4230, 2232, 8865, 2269, 3294, 10231, 5933, 2213, 3375, 4968, 6294, 2743, 11849, 17007, 3479, 5070, 9204, 5605, 6392, 19586, 2318, 4037, 3420, 5957, 5294, 5472, 4244, 7491, 6134, 7823, 13757, 2107, 3441, 3591, 8686, 5473, 4087, 2821, 2851, 5249, 9094, 5324, 6796, 1859]
}
df = pd.DataFrame(data)

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
st.subheader("Model Evaluation")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Visualisierung erstellen
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='Data Points')  # Scatterplot
ax.plot(X, model.predict(X), color='red', label='Linear Regression')  # Regressionslinie
ax.set_xlabel("Job Satisfaction")
ax.set_ylabel("Monthly Income")
ax.set_title("Linear Regression: Job Satisfaction vs. Monthly Income")
ax.legend()

# Plot in Streamlit anzeigen
st.pyplot(fig)

