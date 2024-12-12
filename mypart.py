
# Machine Learning Page and title 
elif st.session_state.page == "Machine Learning":
    st.title("Machine Learning")
    # Button for the homepage
    if st.button("Homepage"):
        # Updates the website back to home
        st.session_state.page = "Home"

    # Trying to extract the necessary and relevant Data from the DataFrame
    try:
        # Relevant columns needed for the regression via Dropna
        # 1. Dropna selects only the columns of the df
        # 2. Ensures that only rows will be used, which have all the values
        df_ml = df[['TotalWorkingYears', 'JobLevel', 'MonthlyIncome']].dropna()
        # Key Error if any relevant column is missing, means Total Woriking years, Job Level, or Monthly Income
    except KeyError as e:
        # Shows the Missing columns
        st.error(f"Missing columns: {e}")
        # If you use the homebutton you will come back to the homepage and the currently executed code will  stop running
        st.stop()
        
    # Importing the necessary libraries for the ML 
    from sklearn.model_selection import train_test_split # for splitting up the data into Trainings- and Testdata
    from sklearn.linear_model import LinearRegression # The model, which is used for the regression
    from sklearn.metrics import mean_squared_error, r2_score # This library was used to check if the Multiple regression is valid. You could delete this part, but it was left in to show that we have checked the model for its validity 
    import numpy as np #for numerical calculations

    # Definition of the input for the regression total working years and job level
    X = df_ml[['TotalWorkingYears', 'JobLevel']]
    # Definition of the output for the regression Monthly Income
    y = df_ml['MonthlyIncome']

    # Splitting up the Traings- and Testdata, while having (automatically, since Tesdata is 30%) 70% Trainings data and 30% Testdata. The random_state ensures that reproducibility, by setting a fixed seed for the random number generator
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Initializing the model 
    model = LinearRegression()
    # With .fit the model "learns" from the data and produces the best fit plane
    # After this line of code the model is trained and the paramters are saved in the model
    model.fit(X_train, y_train)
    # Use the trained model to make predictions with the test dataset (X_test)
    # It applies the learned parameters from above (X_train, y_train) to the input features in X_test to predict/calculate y_pred
    # This is the final step to predict the Monthly Income 
    y_pred = model.predict(X_test)
    # Subheader for the MonthlyIncome prediction 
    st.subheader("Predict Monthly Income")
    
    # Userinput of working years and Joblevel with defined minimum and maximum value and pre selected values in the selection. The User can change the values in this range.
    # Using number_input method for entering the values by the user
    user_working_years = st.number_input("Enter Total Working Years:", min_value=0, max_value=40, step=1, value=10)
    user_job_level = st.number_input("Enter Job Level:", min_value=1, max_value=5, step=1, value=2)
    # Creating a DataFrame for the user input of the working years and the job level via pandas
    # The Total Working and the Job Level are the columns and the user inputs are the rows
    user_input = pd.DataFrame({'TotalWorkingYears': [user_working_years], 'JobLevel': [user_job_level]})
    # The trained model is finally used to predict the Monthly income with the user input values
    # The method .predict does always an Array even if there is just one prediction => [0] ensures that the first and in this case only value will be taken
    predicted_income = model.predict(user_input)[0]
    # Showing the predicet Monthly income in $ with two decimal places
    st.write(f"Predicted Monthly Income: $ *{predicted_income:.2f}*")
    
    # Creating a figure and axis with the sizes 10x6 for a good Overview
    # fig represent the entire figure and ax represents the specific subplot area within the same figure
    fig, ax = plt.subplots(figsize=(10, 6))
    # Creating a scatterplot on the ax from above
    # X-axis represents the total working years from the test data
    # y-axis represents the prediction of the monthly income
    # With virdis a clormap is provided and represents the Joblevels in different colors
    # Virdis is easy to use compared to define our own colors
    # S is the point size and alpha the transperancy
    scatter = ax.scatter(X_test['TotalWorkingYears'], y_pred, c=X_test['JobLevel'], cmap='viridis', s=50, alpha=0.8)
    # Labeling the x and y-axis 
    ax.set_xlabel("Total Working Years")
    ax.set_ylabel("Predicted Monthly Income")
    # Title for scatterplot above the x-axis of the scatterplot 
    ax.set_title("Predicted Monthly Income vs. Total Working Years and Job Level")
    # Adding a color bar next to the scatterplot to show the job level with colormaping
    cbar = plt.colorbar(scatter, ax=ax)
    # Labeling the colorbar with Job Level 
    cbar.set_label("Job Level")
    # Showing a dot on the scatterplot with x-axis User input working years and y axis predicted monthly income 
    # Color red size 100 and zorder of 5 so the dot is visible above the other dot
    # Labeled as your input
    ax.scatter(user_working_years, predicted_income, color='red', s=100, label='Your Input', zorder=5)
    # Visualization of the scatterplot in streamlit
    st.pyplot(fig)

# Employee Report Page with title
elif st.session_state.page == "Employee Report":
    st.title("Employee Report")

    # Loads the OpenAI API Key from the secrets in Streamlit 
    try:
        openai.api_key = st.secrets["openai"]["api_key"]
    # If this is not possible it shows the error down below
    except KeyError:
        st.error("OpenAI API Key is missing. Please check your Streamlit secrets.")
        # Button for Homepage
        if st.button("Homepage"):
            # Updates the website back to home
            st.session_state.page = "Home"
        # If you use the homebutton you will come back to the homepage and the try-except code will  stop running
        st.stop()

    # Checking if the EmployeeNumber column is present in the google sheets data from our API. 
    # Although it is expected to always be present, this check is added for safety.
    if "EmployeeNumber" not in df.columns:
        st.error("The 'EmployeeNumber' column is missing from the Google Sheet Data.")
        # If the user uses the homebutton you will come back to the homepage and this part of the code will stop running    
        if st.button("Homepage"):
            st.session_state.page = "Home"
        # If you use the homebutton you will come back to the homepage and the currently executed code will  stop running
        st.stop()
    # Creating a dropdown menu for the user to select an employee based on their EmployeeNumber.
    # The options in the dropdown are taken from the "EmployeeNumber" column of the DataFrame.
    # The selected value is stored in the variable "employee_number" for further processing.
    # The method .unique() makes sure that the dropdown contains only unique employee numbers.
    employee_number = st.selectbox("Choose Employee (EmployeeNumber)", df["EmployeeNumber"].unique())

    # Filter data for selected employee from the user and using the selected employee data for the dataframe
    employee_data = df[df["EmployeeNumber"] == employee_number]
     #If the data is empty (to check we used the .empty method) it show the error down below
    if employee_data.empty:
        st.error("The selected EmployeeNumber is not present in the data table.")
        # If the user uses the homebutton you will come back to the homepage and this part of the code will stop running    
        if st.button("Homepage"):
            st.session_state.page = "Home"
        # If you use the homebutton you will come back to the homepage and the currently executed code will  stop running
        st.stop()
    # Extracting the selected entry from the filtered employee_data, while making sure that only the values from first row will be chosen via iloc[]
    employee_data = employee_data.iloc[0]

    #Defining the function used for the report
    # This function is responsible for creating a personalized report for an employee based on the provided data
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
            f"Total Working Years: {employee['TotalWorkingYears']}\n" #Using the total amount of working years in general the employee has
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

    # Defines a function to create a PDF report for an employee
    # This function generates a PDF document containing a formatted employee report using the FPDF library
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
         # The report  is used with a normal style ,Arial, not bold and the font size is set to 12, since it is not a title
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
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            # Creating a download button in streamlit which shows up with the emoji and text "Download PDF" 
            # it uses the data from pdf_bytes
            # Naming the File as EmployeeNumber_1.._Report.pdf
            # mime gives the datatype, in this case pdf, so that the data will be correctly used by the device used from the User 
            st.download_button(
                label="📄 Download PDF",
                data=pdf_bytes,
                file_name=f"EmployeeNumber_{employee_data['EmployeeNumber']}_Report.pdf",
                mime="application/pdf"
            )

    # Add buttons for navigation back to homepage
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Homepage"):
            st.session_state.page = "Home"
    # Add buttons for navigation back to Dashboard
    with col2:
        if st.button("Go to Dashboard"):
            st.session_state.page = "Dashboard"

