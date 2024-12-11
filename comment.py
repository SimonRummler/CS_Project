# Employee Report Page with title
elif st.session_state.page == "Employee Report":
    st.title("Employee Report")

    # Tries to Load the OpenAI API Key from the secrets in Streamlit 
    # As source on how to implement open AI API https://www.youtube.com/watch?v=YVFWBJ1WVF8
    try:
        openai.api_key = st.secrets["openai"]["api_key"]
        
    # If this is not possible it shows the error down below
    except KeyError:
        st.error("OpenAI API Key is missing. Please check your Streamlit secrets.")
         # If you use the homebutton you will come back to the homepage and the try-except code will  stop running
        if st.button("Homepage"):
            st.session_state.page = "Home"
        st.stop()

    # Checking if the EmployeeNumber column is present in the google sheets data from our API. The column will always be there, but the code is implemented for additional safety reasons 
    if "EmployeeNumber" not in df.columns:
        st.error("The 'EmployeeNumber' column is missing from the Google Sheet Data.")   
        # If the user uses the homebutton you will come back to the homepage and this part of the code will stop running    
        if st.button("Homepage"):
            st.session_state.page = "Home"
        st.stop()

    # Creating a dropdown menu for the user to select an employee based on their EmployeeNumber.
    # The options in the dropdown are taken from the "EmployeeNumber" column of the DataFrame.
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

    # Extracting the selected entry from the filtered employee_data list, since there will be only one entry per number there is only one column
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
            #Final instructions for the OpenAI/ChatGPT and again making sure that ChatGPT is not going to use any additional information
            "Please create a single paragraph that uses only these details, maintains a professional and formal tone, " 
            "and does not introduce any additional information beyond what is provided."
        )
        # Asking the OpenAi API/ChatGPT, while using the version of chat gpt 3.5
        # Adding necessary attributes to the message: 
            # Defining a role is mandatory for the OpenAI API. In this case, the role is set to "user" because the prompt represents input from the user
            # Limiting the usage of the tokens, which are used for each report. The tokens are purchased at the OpenAI API website
            # Defining a temperature of 0.0, which means that the model works very deterministically and always gives the most probable answer without adding randomness
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.0
            )
            # Returning the first answer generated by the Open AI API/ChatGPT via .choices[0]
            # While using the parameters of message as content. For this code only the content is relevant the role is just a requirement for the OpenAI API as explained
            # And making sure that everything is well formated via .strip()
            return response.choices[0].message["content"].strip()

        # Handle any exceptions that occur during the API request
        # If an error occurs a detailed error message is displayed using st.error(). and None will be returned, so no report will be available
        except Exception as e:
            st.error(f"An error occurred while generating the report: {e}")
            return None

    def create_pdf(report, employee):
        # Creating a new PDF document using the FPDF library assigning, source https://www.youtube.com/watch?v=q70xzDG6nls&list=PLjNQtX45f0dR9K2sMJ5ad9wVjqslNBIC0
        # This line creates a new instance of the FPDF class from the fpdf library, which represents a blank PDF document. The instance is assigned to the variable 'pdf'
        pdf = FPDF()
        #Adding one page to write our report
        pdf.add_page()
        # Set the font for the title of the report
        # Arial is used as the font, "B" indicates bold, and 16 is the font size
        pdf.set_font("Arial", "B", 15)
        # Add the title of the report to the PDF
        # Attributes:
            # width: 0 (spans the whole page)
            # height: 10 (cell height)
            # txt: The title of text, which includes the EmployeeNumber
            # ln: starts a new line after the Title
            # align: C means that it centers the text horizontally
        pdf.cell(0, 10, txt=f"Employee Report (EmployeeNumber: {employee['EmployeeNumber']})", ln=True, align="C")
        # Add vertical spacing after the title, and the new line is implemented 10 units after the title
        pdf.ln(10)
        # The report Arial is used with a normal style not bold or italic and the font size is set to 12, since it is not a title
        pdf.set_font("Arial", size=11)
        # Add the report text to the PDF as multi-line content
        # Parameters:
            # width: 0 (spans the entire width of the page)
            # height: 10 (line height for each row)
            # txt: Uses the report text from the OpenAI API/ Chat GPT
            # multi_cell automatically wraps the text to fit within the page width
        pdf.multi_cell(0, 10, txt=report)
        # It returns and displays the pdf, which has the ability to be displayed and downloaded
        return pdf
        
    # Check if the "Generate Report" button is clicked by the user in the Streamlit app.
    if st.button("Generate Report"):
        # Call the generate_report function, passing the selected employee's data (employee_data) as input. The function returns the report generated from the OpenAi APi/ Chat GPT.
        report_text = generate_report(employee_data)
        # Check if the generated report is not empty or None, which ensures that the report is displayed only if it was generated
        if report_text:
            # Adding a subheader in the Streamlit app "Employee Report:" 
            st.subheader("Employee Report:")
            # Display the generated employee report from OpenAI API/ ChatGPT in the Streamlit app
            st.write(report_text)

            # Call the create_pdf function to generate a PDF report using the generated report and the employee data. The resulting PDF object is assigned to the variable pdf
            pdf = create_pdf(report_text, employee_data)
            # Convert the PDF object into a byte stream, while using the output method of the FPDF library
            # This is necessary in streamlit to create a better ability to download it.
            # The parameter dest=S says that the output is returned as a string 
            # This string is  encoded in latin-1 to make it compatible with PDF standards, so it is downloadable. In this case latin -1 is usefull, because no graphics or emojies are shown in the text document
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            # Creating a download button in streamlit which shows up with the emoji and text "Download PDF" 
            # it uses the data from pdf_bytes
            # Naming the File as EmployeeNumber_Report.pdf
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
