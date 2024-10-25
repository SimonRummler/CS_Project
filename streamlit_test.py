import gspread
from google.oauth2.service_account import Credentials
import streamlit as st

# Dienstkonto-Informationen aus Streamlit Secrets laden
service_account_info = st.secrets["gcp_service_account"]
creds = Credentials.from_service_account_info(service_account_info)

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
