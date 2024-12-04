import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Titel der App
st.title("KNN Classifier: PercentSalaryHike vs. PerformanceRating")

# CSV-Datei laden
csv_file = "HR_Dataset_Group4.5.csv"  # Lokale Datei im selben Verzeichnis
try:
    df = pd.read_csv(csv_file, sep=";")
except FileNotFoundError:
    st.error("CSV-Datei nicht gefunden. Bitte stelle sicher, dass sie im Verzeichnis liegt.")
    st.stop()

# Daten validieren
try:
    df = df[['PercentSalaryHike', 'PerformanceRating']].dropna()
except KeyError:
    st.error("Die Spalten 'PercentSalaryHike' und 'PerformanceRating' fehlen in der Datei.")
    st.stop()

# Daten skalieren
scaler = MinMaxScaler()
df['PercentSalaryHike'] = scaler.fit_transform(df[['PercentSalaryHike']])

# Features und Zielvariable definieren
X = df[['PercentSalaryHike']]
y = df['PerformanceRating']

# Split in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# kNN-Modelleinstellungen
k = 5  # Festgelegter Wert
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Vorhersagen und Bewertung
y_pred = knn.predict(X_test)
st.write("### Modellbewertung")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix anzeigen
st.write("### Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
st.pyplot(fig)

# Benutzerdefinierte Vorhersagen
st.write("### Vorhersage für benutzerdefinierte Eingabe")
user_input = st.number_input("Gib PercentSalaryHike ein (normalisiert zwischen 0 und 1):", min_value=0.0, max_value=1.0, step=0.01)
if user_input:
    prediction = knn.predict([[user_input]])
    st.write(f"Vorhergesagte PerformanceRating: {prediction[0]}")

