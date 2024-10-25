import gspread
from google.oauth2.service_account import Credentials
import streamlit as st

# Pfad zur Dienstkonto-JSON-Datei
SERVICE_ACCOUNT_FILE = 'your-service-account-file.json'

# Google Sheets API-Scopes (Berechtigungen)
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# Anmeldedaten aus der JSON-Datei laden
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# Google Sheets Client erstellen
client = gspread.authorize(creds)

# Google Sheet öffnen (Name des Sheets: "HR-Data")
spreadsheet = client.open('HR-Data')

# Wähle das erste Blatt (Sheet1) aus
sheet = spreadsheet.sheet1

# Alle Daten abrufen
data = sheet.get_all_records()

# Daten in der Streamlit-App anzeigen
st.write("Daten aus dem Google Sheet:")
st.write(data)
