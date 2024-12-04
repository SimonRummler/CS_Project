import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

# Titel der App
st.title("KNN Classifier: Predict PerformanceRating with Adjustable PercentSalaryHike")

# CSV-Datei laden
csv_file = "HR_Dataset_Group4.5.csv"  # Lokale Datei im selben Verzeichnis
try:
    df = pd.read_csv(csv_file, sep=";")
except FileNotFoundError:
    st.error("CSV-Datei nicht gefunden. Bitte stelle sicher, dass sie im Verzeichnis liegt.")
    st.stop()

# Daten validieren
try:
    df = df[['EmployeeNumber', 'PercentSalaryHike', 'PerformanceRating']].dropna()
except KeyError:
    st.error("Die Spalten 'EmployeeNumber', 'PercentSalaryHike' und 'PerformanceRating' fehlen in der Datei.")
    st.stop()

# Daten skalieren
scaler = MinMaxScaler()
df['PercentSalaryHike_scaled'] = scaler.fit_transform(df[['PercentSalaryHike']])

# Features und Zielvariable definieren
X = df[['PercentSalaryHike_scaled']]
y = df['PerformanceRating']

# Split in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# kNN-Modelleinstellungen
k = 5  # Festgelegter Wert
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Eingabefeld für EmployeeNumber
st.write("### Vorhersage des PerformanceRatings basierend auf EmployeeNumber")
employee_number = st.number_input("Gib die EmployeeNumber ein:", min_value=int(df['EmployeeNumber'].min()), max_value=int(df['EmployeeNumber'].max()), step=1)

if employee_number in df['EmployeeNumber'].values:
    # Zeige aktuelle PercentSalaryHike
    current_percent_hike = df.loc[df['EmployeeNumber'] == employee_number, 'PercentSalaryHike'].values[0]
    st.write(f"Aktueller PercentSalaryHike für EmployeeNumber {employee_number}: {current_percent_hike}%")

    # Eingabefeld für neuen PercentSalaryHike
    new_percent_hike = st.slider("Ändere den PercentSalaryHike-Wert:", min_value=0, max_value=100, value=int(current_percent_hike), step=1)

    # Skaliere den neuen Wert
    new_percent_hike_scaled = scaler.transform([[new_percent_hike]])[0][0]

    # Vorhersage basierend auf dem neuen PercentSalaryHike
    prediction = knn.predict([[new_percent_hike_scaled]])[0]
    st.write(f"Vorhergesagtes PerformanceRating für EmployeeNumber {employee_number} mit PercentSalaryHike {new_percent_hike}%: {prediction}")
else:
    st.error("Die eingegebene EmployeeNumber ist nicht in den Daten enthalten.")

