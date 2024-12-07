import streamlit as st
import pandas as pd
from fpdf import FPDF
import openai
import io

# OpenAI API-Key aus Streamlit Secrets
try:
    openai.api_key = st.secrets["openai"]["api_key"]
except KeyError:
    st.error("OpenAI API Key is missing. Please check your Streamlit secrets.")
    st.stop()

# CSV-Datei laden
try:
    data = pd.read_csv("12342.csv", delimiter=";")
except FileNotFoundError:
    st.error("Die CSV-Datei '12342.csv' wurde nicht gefunden. Bitte stelle sicher, dass sie im selben Verzeichnis liegt.")
    st.stop()

# Prüfung auf UID-Spalte
if "UID" not in data.columns:
    st.error("Die Spalte 'UID' fehlt in der CSV-Datei. Bitte überprüfe die Daten.")
    st.stop()

# Auswahlbox für Mitarbeiter
uid = st.selectbox("Mitarbeiter auswählen (UID)", data["UID"].unique(), key="select_uid")

# Daten des ausgewählten Mitarbeiters laden
employee_data = data[data["UID"] == uid]
if employee_data.empty:
    st.error("Die ausgewählte UID ist nicht in der Datentabelle vorhanden.")
    st.stop()

employee_data = employee_data.iloc[0]  # Ersten Eintrag nehmen

# Funktion zur Berichtserzeugung
def generate_report(employee):
    try:
        prompt = (
            f"Write a professional employee report based on the following details:\n\n"
            f"Age: {employee['Age']}\n"
            f"Department: {employee['Department']}\n"
            f"Business Travel: {employee['BusinessTravel']}\n"
            f"Years at Company: {employee['YearsAtCompany']}\n"
            f"Education Field: {employee['EducationField']}\n"
            f"Work-Life Balance: {employee['WorkLifeBalance']}\n"
            f"Total Working Years: {employee['TotalWorkingYears']}\n\n"
            "Please write the report in a formal tone suitable for business use."
        )
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"Beim Erstellen des Berichts ist ein Fehler aufgetreten: {e}")
        return None

# PDF-Erstellungsfunktion
def create_pdf(report, employee):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, txt=f"Employee Report (UID: {employee['UID']})", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=report)
    return pdf

# Bericht generieren und anzeigen
if st.button("Generate Report"):
    report_text = generate_report(employee_data)
    if report_text:
        st.subheader("Employee Report:")
        st.write(report_text)

        # PDF generieren und zum Download anbieten
        pdf = create_pdf(report_text, employee_data)
        pdf_bytes = pdf.output(dest='S').encode('latin-1')  # PDF als Bytes im Speicher
        st.download_button(
            label="📄 Download PDF",
            data=pdf_bytes,
            file_name=f"UID_{employee_data['UID']}_Report.pdf",
            mime="application/pdf"
        )


