import streamlit as st
import pandas as pd
from fpdf import FPDF
import openai
import io

def main():
    # OpenAI API-Key from Streamlit Secrets
    try:
        openai.api_key = st.secrets["openai"]["api_key"]
    except KeyError:
        st.error("OpenAI API Key is missing. Please check your Streamlit secrets.")
        st.stop()

    # Load CSV file
    try:
        data = pd.read_csv("12342.csv", delimiter=";")
    except FileNotFoundError:
        st.error("The CSV file '12342.csv' was not found. Please ensure it is in the same directory.")
        st.stop()

    # Check for UID column
    if "UID" not in data.columns:
        st.error("The 'UID' column is missing from the CSV file. Please verify your data.")
        st.stop()

    # Employee selection
    uid = st.selectbox("Choose Employee (UID)", data["UID"].unique(), key="select_uid_widget")

    # Load selected employee data
    employee_data = data[data["UID"] == uid]
    if employee_data.empty:
        st.error("The selected UID is not present in the data table.")
        st.stop()

    employee_data = employee_data.iloc[0]

    # Generate report using ChatCompletion
    def generate_report(employee):
        try:
            # Build the prompt using the provided data fields
            prompt = (
                "Create a short, formal, and professional employee report in English using only the provided data. "
                "The employee does not have a name, so please refer to them by their UID. "
                "Do not add any information not present in the data. Present the information as a cohesive paragraph "
                "without additional speculation.\n\n"
                f"Data:\n"
                f"UID: {employee['UID']}\n"
                f"Age: {employee['Age']}\n"
                f"Department: {employee['Department']}\n"
                f"Job Role: {employee['JobRole']}\n"
                f"Gender: {employee['Gender']}\n"
                f"Education Field: {employee['EducationField']}\n"
                f"Years at Company: {employee['YearsAtCompany']}\n"
                f"Total Working Years: {employee['TotalWorkingYears']}\n"
                f"Monthly Income: {employee['MonthlyIncome']}\n"
                f"Business Travel: {employee['BusinessTravel']}\n"
                f"Overtime: {employee['OverTime']}\n"
                f"Job Satisfaction (1–4): {employee['JobSatisfaction']}\n"
                f"Work-Life Balance (1–4): {employee['WorkLifeBalance']}\n"
                f"Relationship Satisfaction (1–4): {employee['RelationshipSatisfaction']}\n"
                f"Performance Rating: {employee['PerformanceRating']}\n"
                f"Training Times Last Year: {employee['TrainingTimesLastYear']}\n\n"
                "Please create a single paragraph that uses only these details, maintains a professional and formal tone, "
                "and does not introduce any additional information beyond what is provided."
            )

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.0
            )
            return response.choices[0].message["content"].strip()

        except Exception as e:
            st.error(f"An error occurred while generating the report: {e}")
            return None

    # PDF creation function
    def create_pdf(report, employee):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, txt=f"Employee Report (UID: {employee['UID']})", ln=True, align="C")
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=report)
        return pdf

    # Generate and display report
    if st.button("Generate Report", key="generate_report_button"):
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


