import streamlit as st
import pandas as pd
from fpdf import FPDF
import requests
import io

def main():
    # Hugging Face API-Key aus Streamlit Secrets laden
    try:
        hf_token = st.secrets["hf"]["api_token"]
    except KeyError:
        st.error("Hugging Face API Token is missing. Please add it to your Streamlit secrets.")
        st.stop()

    # Hugging Face Inference API für Falcon 7B Instruct (Beispiel) nutzen
    API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    headers = {"Authorization": f"Bearer {hf_token}"}

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

    # Mitarbeiter per Dropdown auswählen
    uid = st.selectbox("Mitarbeiter auswählen (UID)", data["UID"].unique(), key="select_uid_widget")

    # Daten des ausgewählten Mitarbeiters holen
    employee_data = data[data["UID"] == uid]
    if employee_data.empty:
        st.error("Die ausgewählte UID ist nicht in der Datentabelle vorhanden.")
        st.stop()

    employee_data = employee_data.iloc[0]

    # Funktion zur Berichtserzeugung über Hugging Face Inference API
    def generate_report(employee):
        try:
            prompt = (
                "Write a professional employee report based on the following details:\n\n"
                f"Age: {employee['Age']}\n"
                f"Department: {employee['Department']}\n"
                f"Business Travel: {employee['BusinessTravel']}\n"
                f"Years at Company: {employee['YearsAtCompany']}\n"
                f"Education Field: {employee['EducationField']}\n"
                f"Work-Life Balance: {employee['WorkLifeBalance']}\n"
                f"Total Working Years: {employee['TotalWorkingYears']}\n\n"
                "Please write the report in a formal tone suitable for business use."
            )

            payload = {"inputs": prompt}
            response = requests.post(API_URL, headers=headers, json=payload)
            result = response.json()

            # Prüfen, ob "generated_text" im Ergebnis enthalten ist
            if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                return result[0]["generated_text"].strip()
            else:
                return "Keine gültige Antwort erhalten."
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

    # Button zum Erstellen des Berichts
    if st.button("Generate Report", key="generate_report_button"):
        with st.spinner("Bericht wird generiert..."):
            report_text = generate_report(employee_data)
        if report_text:
            st.subheader("Employee Report:")
            st.write(report_text)

            pdf = create_pdf(report_text, employee_data)
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button(
                label="📄 Download PDF",
                data=pdf_bytes,
                file_name=f"UID_{employee_data['UID']}_Report.pdf",
                mime="application/pdf",
                key="download_pdf_button"
            )

if __name__ == "__main__":
    main()
