import gspread
from google.oauth2.service_account import Credentials
import streamlit as st

# Pfad zur Dienstkonto-JSON-Datei (passe diesen Pfad an, wenn nötig)
SERVICE_ACCOUNT_FILE = '/mnt/data/round-seeker-439709-p8-b1cf9a6ceb00.json'

# Google Sheets API-Scopes (Berechtigungen) festlegen
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# Anmeldedaten aus der JSON-Datei laden
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# Google Sheets Client erstellen
client = gspread.authorize(creds)

# Google Sheet öffnen (Name des Sheets: "HR-Data")
spreadsheet = client.open('HR-Data')

# Wähle das erste Blatt (Sheet1) aus
sheet = spreadsheet.sheet1

# Alle Daten als Liste von Dictionaries abrufen
data = sheet.get_all_records()

# Die abgerufenen Daten in der Streamlit App anzeigen
st.write("Daten aus dem Google Sheet:")
st.write(data)
