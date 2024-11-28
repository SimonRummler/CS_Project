# Import der notwendigen Bibliotheken
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Beispiel-Daten: Performance und Gehalt
data = {
    "Performance": [3.8, 4.2, 4.5, 4.0, 3.6, 4.8, 3.9],
    "Salary": [50000, 55000, 60000, 52000, 48000, 65000, 51000]
}
df = pd.DataFrame(data)

# Features (Performance) und Zielvariable (Gehalt)
X = df[["Performance"]]
y = df["Salary"]

# Aufteilen der Daten in Trainings- und Testdatensätze
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisieren und Trainieren des Modells
model = LinearRegression()
model.fit(X_train, y_train)

# Vorhersagen für Testdaten
y_pred = model.predict(X_test)

# Modellbewertung
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Ergebnisse in der Konsole ausgeben
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Visualisierung erstellen
plt.scatter(X, y, color='blue', label='Datenpunkte')  # Scatterplot der Daten
plt.plot(X, model.predict(X), color='red', label='Lineare Regression')  # Regressionslinie
plt.xlabel('Performance-Bewertung')
plt.ylabel('Gehalt (in €)')
plt.title('Lineare Regression: Performance vs. Gehalt')
plt.legend()
plt.grid(True)

# Plot als Bild speichern
plt.savefig('linear_regression_plot.png')  # Bild speichern für GitHub
plt.show()
