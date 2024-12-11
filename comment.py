# Employee Report Page
elif st.session_state.page == "Employee Report":
    st.title("Employee Report")

    # Tries to Load the OpenAI API Key from the secrets in Streamlit
    try:
        openai.api_key = st.secrets["openai"]["api_key"]
        
    # If this is not possible it shows the error down below
    except KeyError:
        st.error("OpenAI API Key is missing. Please check your Streamlit secrets.")
         # If you use the homebutton you will come back to the homepage and the try-except code will  stop running
        if st.button("Homepage"):
            st.session_state.page = "Home"
        st.stop()

    # Checking if the EmployeeNumber column is present in the google sheets data from our API. The column will always be there, but the code is implemented for additional safety reasons. 
    if "EmployeeNumber" not in df.columns:
        st.error("The 'EmployeeNumber' column is missing from the Google Sheet Data.")   
        # If the user uses the homebutton you will come back to the homepage and this part of the code will stop running    
        if st.button("Homepage"):
            st.session_state.page = "Home"
        st.stop()

    # Creating a dropdown menu for the user to select an employee based on their EmployeeNumber.
    # The options in the dropdown are populated from the "EmployeeNumber" column of the DataFrame.
    # The selected value is stored in the variable "employee_number" for further processing.
    employee_number = st.selectbox("Choose Employee (EmployeeNumber)", df["EmployeeNumber"])

    # Filter data for selected employee from the user and using the selected employee data for the dataframe
    employee_data = df[df["EmployeeNumber"] == employee_number]

    #If the data is empty (to check we used the .empty method) it show the error down below
    if employee_data.empty:
        st.error("The selected EmployeeNumber is not present in the data table.")
        # If the user uses the homebutton you will come back to the homepage and this part of the code will stop running    
        if st.button("Homepage"):
            st.session_state.page = "Home"
        st.stop()

    # Extracting the selected entry from the filtered employee_data list
    employee_data = employee_data[0]
    
    def generate_report(employee):
        # Build the prompt for the OpenAI API ChatGPT connection, while only using the given data
        prompt = (
            "Create a short, formal, and professional employee report in English using only the provided data. "
            "The employee does not have a name, so please refer to them by their EmployeeNumber. "
            "Do not add any information not present in the data. Present the information as a cohesive paragraph "
            "without additional speculation.\n\n"
            # The prompt uses the data from the selected employee (number), see down below
            f"Data:\n" #Label to begin here 
            f"EmployeeNumber: {employee['EmployeeNumber']}\n" #Using the EmployeeNumber
            f"Age: {employee['Age']}\n" #Using the Age of the employee
            f"Department: {employee['Department']}\n" #Using the department, in which the employee works
            f"Job Role: {employee['JobRole']}\n" #Using the Job role the employee has
            f"Gender: {employee['Gender']}\n" #Using the Gender the employee has
            f"Education Field: {employee['EducationField']}\n" #Using the Education field the employee has
            f"Years at Company: {employee['YearsAtCompany']}\n" #Using the total amount of working years at the company the employee has
            f"Total Working Years: {employee['TotalWorkingYears']}\n"  #Using the total amount of working years in general the employee has
            f"Monthly Income: {employee['MonthlyIncome']}\n" #Using the monthly income the employee has
            f"Business Travel: {employee['BusinessTravel']}\n" #Using the data if the employee is traveling or not
            f"Overtime: {employee['OverTime']}\n" #Using if the employee has Over time
            f"Job Satisfaction (1–4): {employee['JobSatisfaction']}\n" #Using the Job satisfaction of the employee
            f"Work-Life Balance (1–4): {employee['WorkLifeBalance']}\n" #Using the Work life balance of the employee
            f"Relationship Satisfaction (1–4): {employee['RelationshipSatisfaction']}\n" #Using the relationship satisfaction of the employee
            f"Performance Rating: {employee['PerformanceRating']}\n" #Using the performance rating of the employee
            f"Training Times Last Year: {employee['TrainingTimesLastYear']}\n\n" #Using the training times of the employee
            #Final instructions for the OpenAI/ChatGPT adn again making sure that the LLM is not going to use any additional information
            "Please create a single paragraph that uses only these details, maintains a professional and formal tone, " 
            "and does not introduce any additional information beyond what is provided."
        )

        try:
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

    def create_pdf(report, employee):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, txt=f"Employee Report (EmployeeNumber: {employee['EmployeeNumber']})", ln=True, align="C")
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=report)
        return pdf

    if st.button("Generate Report"):
        report_text = generate_report(employee_data)
        if report_text:
            st.subheader("Employee Report:")
            st.write(report_text)

            pdf = create_pdf(report_text, employee_data)
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button(
                label="📄 Download PDF",
                data=pdf_bytes,
                file_name=f"EmployeeNumber_{employee_data['EmployeeNumber']}_Report.pdf",
                mime="application/pdf"
            )

    # Add buttons for navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Homepage"):
            st.session_state.page = "Home"
    with col2:
        if st.button("Go to Dashboard"):
            st.session_state.page = "Dashboard"
