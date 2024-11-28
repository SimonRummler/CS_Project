import streamlit as st
import pandas as pd
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Google API Credentials aus Streamlit Secrets laden
credentials_info = st.secrets["Versuch"]

# Service Account Credentials erstellen
credentials = service_account.Credentials.from_service_account_info(
    json.loads(credentials_info)
)

# Verbindung zur Google Sheets API herstellen
service = build('sheets', 'v4', credentials=credentials)

# Google Sheets Konfiguration
SPREADSHEET_ID = "19CC438qwcEpCufbyukbzQ1RmVW9uZ1VK6rFHtXNU8IU"  # Spreadsheet ID
RANGE_NAME = "sheet1"  # Der gesamte Sheet-Name

# Daten aus Google Sheets abrufen
sheet = service.spreadsheets()
result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
values = result.get("values", [])

# Daten in einen DataFrame umwandeln
if not values:
    st.error("Keine Daten im Google Sheet gefunden!")
else:
    # Erste Zeile als Header verwenden
    df = pd.DataFrame(values[1:], columns=values[0])  # Daten ab Zeile 2, Header ist Zeile 1

    # Konvertiere numerische Spalten (falls notwendig)
    if "Performance" in df.columns and "Salary" in df.columns:
        df["Performance"] = pd.to_numeric(df["Performance"], errors="coerce")
        df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")

        # Filtere gültige Zeilen (ohne NaN-Werte)
        df = df.dropna(subset=["Performance", "Salary"])

        st.write("Daten aus Google Sheets:")
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

        # Visualisierung
        fig, ax = plt.subplots()
        ax.scatter(X, y, color='blue', label='Datenpunkte')  # Scatterplot
        ax.plot(X, model.predict(X), color='red', label='Lineare Regression')  # Regressionslinie
        ax.set_xlabel("Performance")
        ax.set_ylabel("Salary")
        ax.set_title("Lineare Regression: Performance vs. Salary")
        ax.legend()

        # Plot in Streamlit anzeigen
        st.pyplot(fig)
    else:
        st.error("Die benötigten Spalten 'Performance' und 'Salary' wurden nicht gefunden!")
