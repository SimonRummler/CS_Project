import streamlit as st
import pandas as pd
from fpdf import FPDF
import openai

# OpenAI API-Key aus Streamlit Secrets
openai.api_key = st.secrets["openai"]["api_key"]

# CSV-Datei laden
data = pd.read_csv("12342.csv", sep=";")  # Datentrenner ist hier ein Semikolon

# Debugging: Verfügbare Spalten anzeigen
st.write("Available columns in the CSV:", data.columns.tolist())

# Sicherstellen, dass die Spalte 'UID' existiert
if "UID" not in data.columns:
    st.error("The column 'UID' does not exist in the CSV file. Please check the file.")
    st.stop()

# Mitarbeiter auswählen
uid = st.selectbox("Select Employee UID", data["UID"])

# Daten des ausgewählten Mitarbeiters
employee_data = data[data["UID"] == uid].iloc[0]

# Berichtsgenerierungsfunktion
def generate_report(employee):
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

# Bericht generieren und anzeigen
if st.button("Generate Report"):
    report_text = generate_report(employee_data)
    st.subheader("Employee Report:")
    st.write(report_text)

    # PDF-Erstellungsfunktion
    def create_pdf(report, employee):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Employee Report for UID {employee['UID']}", ln=True, align="C")
        pdf.ln(10)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 10, txt=report)
        return pdf

    # PDF generieren und Download ermöglichen
    pdf = create_pdf(report_text, employee_data)
    pdf_output = f"UID_{employee_data['UID']}_Report.pdf"
    pdf.output(pdf_output)

    with open(pdf_output, "rb") as file:
        st.download_button(
            label="📄 Download PDF",
            data=file,
            file_name=pdf_output,
            mime="application/pdf"
        )

