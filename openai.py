import streamlit as st
import pandas as pd
from fpdf import FPDF
import openai

# Debugging: Überprüfen, ob der API-Key geladen wird
try:
    openai.api_key = st.secrets["openai"]["api_key"]
    st.write("OpenAI API Key loaded:", openai.api_key is not None)
except KeyError:
    st.error("OpenAI API Key not found in secrets. Please configure it correctly.")
    st.stop()

# CSV-Datei laden
try:
    data = pd.read_csv("12342.csv", sep=";")  # Verwende das richtige Trennzeichen
    st.write("CSV file loaded successfully.")
except FileNotFoundError:
    st.error("The CSV file '12342.csv' was not found. Please ensure it is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    st.stop()

# Debugging: Verfügbare Spalten in der CSV anzeigen
st.write("Available columns in the CSV:", data.columns.tolist())

# Sicherstellen, dass die erwarteten Spalten existieren
required_columns = [
    "UID", "Age", "Department", "BusinessTravel", "YearsAtCompany",
    "EducationField", "WorkLifeBalance", "TotalWorkingYears"
]
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    st.error(f"The following required columns are missing from the CSV: {missing_columns}")
    st.stop()

# Mitarbeiter auswählen
try:
    uid = st.selectbox("Select Employee UID", data["UID"])
except KeyError:
    st.error("The column 'UID' does not exist in the CSV file.")
    st.stop()

# Daten des ausgewählten Mitarbeiters
try:
    employee_data = data[data["UID"] == uid].iloc[0]
    st.write("Selected employee data:", employee_data)
except IndexError:
    st.error("No data found for the selected UID.")
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
        # Debugging: Prompt anzeigen
        st.write("Generated prompt for OpenAI API:", prompt)

        # OpenAI API-Aufruf
        response = openai.Completion.create(
            model="text-davinci-003",  # Korrektur: "model" statt "engine"
            prompt=prompt,
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"Error generating report: {e}")
        return ""

# Bericht generieren und anzeigen
if st.button("Generate Report"):
    report_text = generate_report(employee_data)
    if report_text:
        st.subheader("Employee Report:")
        st.write(report_text)

        # PDF-Erstellungsfunktion
        def create_pdf(report, employee):
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"Employee Report for UID {employee['UID']}", ln=True, align="C")
                pdf.ln(10)
                pdf.set_font("Arial", size=10)
                pdf.multi_cell(0, 10, txt=report)
                return pdf
            except Exception as e:
                st.error(f"Error creating PDF: {e}")
                return None

        # PDF generieren und Download ermöglichen
        pdf = create_pdf(report_text, employee_data)
        if pdf:
            pdf_output = f"UID_{employee_data['UID']}_Report.pdf"
            pdf.output(pdf_output)

            with open(pdf_output, "rb") as file:
                st.download_button(
                    label="📄 Download PDF",
                    data=file,
                    file_name=pdf_output,
                    mime="application/pdf"
                )
    else:
        st.error("Failed to generate the report. Please check the logs for more details.")
