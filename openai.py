import streamlit as st
import pandas as pd
from fpdf import FPDF
import openai

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
    st.error("The CSV file '12342.csv' was not found. Please ensure it is in the same directory.")
    st.stop()

# Streamlit UI
st.title("Employee Report Generator")

# Verfügbare Spalten überprüfen und anzeigen
if st.checkbox("Show available columns in the CSV", key="show_columns"):
    st.write(data.columns.tolist())

# Mitarbeiter auswählen
if "UID" not in data.columns:
    st.error("The column 'UID' is missing from the dataset. Please check the CSV file.")
    st.stop()

uid = st.selectbox("Select Employee UID", data["UID"], key="unique_uid_selectbox")

# Daten des ausgewählten Mitarbeiters
try:
    employee_data = data[data["UID"] == uid].iloc[0]
except IndexError:
    st.error("The selected UID does not exist in the dataset.")
    st.stop()

# Berichtsgenerierungsfunktion
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
        st.error(f"An error occurred while generating the report: {e}")
        return None

# Bericht generieren und anzeigen
if st.button("Generate Report", key="generate_report_button"):
    report_text = generate_report(employee_data)
    if report_text:
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


