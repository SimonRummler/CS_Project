import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from fpdf import FPDF
import io

# Define scopes
scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# Authenticate using Streamlit secrets
credentials = Credentials.from_service_account_info(
    st.secrets["google_credentials"],
    scopes=scopes
)
client = gspread.authorize(credentials)

# Access Google Sheet
sheet = client.open("Dataset").sheet1
data = sheet.get_all_records()
df = pd.DataFrame(data)

# Initialize session state for page tracking
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Home Page
if st.session_state.page == "Home":
    st.title("Welcome to the HR monitor!")
    st.write("This application provides you with in-depth analysis of your HR data in real time.")
    st.write("Choose a section to navigate to for different functionalities:")
    st.write("- **Dashboard:** Explore an interactive HR dashboard offering comprehensive insights into your workforce, helping you make data-driven decisions.")
    st.write("- **Machine Learning:** Harness the power of AI to predict employee's income.")
    st.write("- **Data Management:** Efficiently manage employee records by adding, editing, or deleting information with ease.")
    st.write("- **Employee Report:** Generate a professional report for a selected employee.")

    # Display navigation buttons under the explanation
    st.write("### Navigate to:")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("Dashboard"):
            st.session_state.page = "Dashboard"
    with col2:
        if st.button("Machine Learning"):
            st.session_state.page = "Machine Learning"
    with col3:
        if st.button("Data Management"):
            st.session_state.page = "Data Management"
    with col4:
        if st.button("Employee Report"):
            st.session_state.page = "Employee Report"


# Dashboard Page
elif st.session_state.page == "Dashboard":
    st.title("Google Sheets Data Analysis")

    # Display 'Homepage' button
    if st.button("Homepage"):
        st.session_state.page = "Home"

     # Income Statistics
    st.subheader("Income Statistics")
    mean_income = df['MonthlyIncome'].mean()
    median_income = df['MonthlyIncome'].median()
    std_income = df['MonthlyIncome'].std()
    st.write(f"Mean Monthly Income: ${mean_income:.2f}")
    st.write(f"Median Monthly Income: ${median_income:.2f}")
    st.write(f"Standard Deviation of Monthly Income: ${std_income:.2f}")

    st.subheader("Multiple Subplots: Age, Income, and Distance from Home")
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    sns.histplot(df['Age'], bins=20, kde=True, ax=axs[0], color='skyblue')
    axs[0].set_title('Age Distribution')
    sns.histplot(df['MonthlyIncome'], bins=20, kde=True, ax=axs[1], color='green')
    axs[1].set_title('Monthly Income Distribution')
    sns.histplot(df['DistanceFromHome'], bins=20, kde=True, ax=axs[2], color='orange')
    axs[2].set_title('Distance from Home Distribution')
    st.pyplot(fig)

    st.subheader("Department, Gender, and Job Role Insights")
    fig, axs = plt.subplots(1, 3, figsize=(25, 8))
    
    department_counts = df['Department'].value_counts()
    percentages = department_counts / department_counts.sum() * 100
    axs[0].pie(percentages, labels=percentages.index, autopct='%1.1f%%', startangle=140)
    axs[0].set_title('Employees by Department', fontsize=14)
    axs[0].axis('equal')
    
    sns.violinplot(x='Gender', y='Age', data=df, hue='Gender', split=True, ax=axs[1])
    axs[1].set_title('Age Distribution by Gender', fontsize=14)
    axs[1].set_xlabel('Gender')
    axs[1].set_ylabel('Age')
    
    sns.boxplot(y='JobRole', x='MonthlyIncome', data=df, ax=axs[2])
    axs[2].set_title('Monthly Income by Job Role', fontsize=14)
    axs[2].set_xlabel('Monthly Income')
    axs[2].set_ylabel('Job Role')
    
    fig.suptitle('Key HR Insights', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)

    st.subheader("Insights on Income, Age, and Department")
    fig, axs = plt.subplots(1, 3, figsize=(25, 8))
    
    sns.kdeplot(data=df['MonthlyIncome'], fill=True, color='skyblue', alpha=0.5, ax=axs[0])
    axs[0].set_title('Density Plot of Monthly Income', fontsize=14)
    axs[0].set_xlabel('Monthly Income')
    axs[0].set_ylabel('Density')
    
    sns.scatterplot(x=df['Age'], y=df['MonthlyIncome'], ax=axs[1], alpha=0.6, color='blue')
    sns.regplot(x='Age', y='MonthlyIncome', data=df, ax=axs[1], scatter=False, color='red')
    axs[1].set_title('Age vs. Monthly Income', fontsize=14)
    axs[1].set_xlabel('Age')
    axs[1].set_ylabel('Monthly Income')
    
    grouped_data = df.groupby(['Department', 'BusinessTravel']).size().unstack(fill_value=0)
    grouped_data.plot(kind='bar', ax=axs[2], stacked=False, color=['lightblue', 'orange', 'green'])
    axs[2].set_title('Department vs. Business Travel', fontsize=14)
    axs[2].set_xlabel('Department')
    axs[2].set_ylabel('Number of Employees')
    axs[2].tick_params(axis='x', rotation=45)
    
    fig.suptitle('Key Insights: Income, Age, and Department', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)

    st.subheader("Department and Business Travel Insights")
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    
    department_counts.plot(kind='bar', color='skyblue', edgecolor='black', ax=axs[0])
    axs[0].set_title('Distribution of Employees by Department', fontsize=14)
    axs[0].set_xlabel('Department')
    axs[0].set_ylabel('Number of Employees')
    axs[0].tick_params(axis='x', rotation=45)
    
    business_travel_counts = df['BusinessTravel'].value_counts()
    business_travel_counts.plot(kind='bar', color='lightcoral', edgecolor='black', ax=axs[1])
    axs[1].set_title('Business Travel Frequency', fontsize=14)
    axs[1].set_xlabel('Business Travel Category')
    axs[1].set_ylabel('Number of Employees')
    axs[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Joint Plot: Age vs. Monthly Income")
    fig = sns.jointplot(x='Age', y='MonthlyIncome', data=df, kind='reg', height=8, space=0.2)
    st.pyplot(fig)


# Machine Learning Page
elif st.session_state.page == "Machine Learning":
    st.title("Machine Learning")
    if st.button("Homepage"):
        st.session_state.page = "Home"

    # Import and model training
    # (Die Daten liegen bereits in df vor)
    try:
        # Relevant columns for regression
        df_ml = df[['TotalWorkingYears', 'JobLevel', 'MonthlyIncome']].dropna()
    except KeyError as e:
        st.error(f"Fehlende Spalten: {e}")
        st.stop()

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    X = df_ml[['TotalWorkingYears', 'JobLevel']]
    y = df_ml['MonthlyIncome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.subheader("Predict Monthly Income")
    user_working_years = st.number_input("Enter Total Working Years:", min_value=0, max_value=40, step=1, value=10)
    user_job_level = st.number_input("Enter Job Level:", min_value=1, max_value=5, step=1, value=2)

    user_input = pd.DataFrame({'TotalWorkingYears': [user_working_years], 'JobLevel': [user_job_level]})
    predicted_income = model.predict(user_input)[0]
    st.write(f"Predicted Monthly Income: *{predicted_income:.2f}*")

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_test['TotalWorkingYears'], y_pred, c=X_test['JobLevel'], cmap='viridis', s=50, alpha=0.8)
    ax.set_xlabel("Total Working Years")
    ax.set_ylabel("Predicted Monthly Income")
    ax.set_title("Predicted Monthly Income vs. Total Working Years (Color: Job Level)")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Job Level")
    ax.scatter(user_working_years, predicted_income, color='red', s=100, label='Your Input', zorder=5)
    ax.legend()
    st.pyplot(fig)

# Backend Page (Data Management)
elif st.session_state.page == "Data Management":
    st.title("Data Input Manager")

    spreadsheet_name = "Dataset"
    worksheet = client.open(spreadsheet_name).sheet1
    headers = worksheet.row_values(1)
    sheet_data = worksheet.get_all_values()

    def main_page():
        st.header("Main Page")
        st.write("Choose an action:")
        if st.button("Add New Employee Data"):
            st.session_state.Backend_subpage = "add_employee"
        if st.button("Change Employee Data"):
            st.session_state.Backend_subpage = "change_employee"
        if st.button("Delete Employee Data"):
            st.session_state.Backend_subpage = "delete_employee"
        if st.button("Homepage"):
            st.session_state.page = "Home"

    def add_employee_page():
        st.header("Add New Employee Data")
        st.write("Enter details for the new employee:")

        employee_numbers = [
            int(row[headers.index("EmployeeNumber")])
            for row in sheet_data[1:]
            if row[headers.index("EmployeeNumber")].isdigit()
        ]
        next_employee_number = max(employee_numbers) + 1

        with st.form("add_employee_form"):
            age = st.selectbox("Age", list(range(18, 63)))
            attrition = st.selectbox("Attrition", ["Yes", "No"])
            business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            daily_rate = st.number_input("Daily Rate", min_value=0, step=1)
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            distance_from_home = st.number_input("Distance from Home (km)", min_value=0, step=1)
            education = st.slider("Education level", min_value=1, max_value=5)
            education_field = st.selectbox("Education Field", [ "Human Resources", "Life Sciences", "Marketing", "Medical", "Technical Degree", "Other"])
            employee_count = st.number_input("Employee Count", min_value=1, step=1)
            employee_number = next_employee_number
            environment_satisfaction = st.slider("Environment Satisfaction", min_value=1, max_value=4)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            hourly_rate = st.number_input("Hourly Rate", min_value=0, step=1)
            job_involvement = st.slider("Job Involvement", min_value=1, max_value=4)
            job_level = st.slider("Job Level", min_value=1, max_value=5)
            job_role = st.selectbox("Job Role", ["Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager", "Manufacturing Director", "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"])
            job_satisfaction = st.slider("Job Satisfaction", min_value=1, max_value=4)
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            monthly_income = st.number_input("Monthly Income ($)", min_value=1009, step=1)
            monthly_rate = st.number_input("Monthly Rate", min_value=1000, step=100)
            num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, step=1)
            over_18 = st.selectbox("Over 18", ["Y", "N"])
            overtime = st.selectbox("Overtime", ["Yes", "No"])
            percent_salary_hike = st.slider("Percent Salary Hike", min_value=0, max_value=100)
            performance_rating = st.slider("Performance Rating", min_value=1, max_value=5)
            relationship_satisfaction = st.slider("Relationship Satisfaction", min_value=1, max_value=4)
            standard_hours = st.number_input("Standard Hours", min_value=1, step=1)
            stock_option_level = st.number_input("Stock Option Level", min_value=0, step=1)
            total_working_years = st.number_input("Total Working Years", min_value=0, step=1)
            training_times_last_year = st.number_input("Training Times Last Year", min_value=0, step=1)
            work_life_balance = st.slider("Work-Life Balance", min_value=1, max_value=4)
            years_at_company = st.number_input("Years at Company", min_value=0, step=1)
            years_in_current_role = st.number_input("Years in Current Role", min_value=0, step=1)
            years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, step=1)
            years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, step=1)

            if st.form_submit_button("Submit"):
                new_employee = [
                    age, attrition, business_travel, daily_rate, department, distance_from_home,
                    education, education_field, employee_count, employee_number, environment_satisfaction,
                    gender, hourly_rate, job_involvement, job_level, job_role, job_satisfaction, marital_status,
                    monthly_income, monthly_rate, num_companies_worked, over_18, overtime, percent_salary_hike,
                    performance_rating, relationship_satisfaction, standard_hours, stock_option_level,
                    total_working_years, training_times_last_year, work_life_balance, years_at_company,
                    years_in_current_role, years_since_last_promotion, years_with_curr_manager
                ]
                worksheet.append_row(new_employee)
                st.success(f"Employee added successfully with Employee Number {employee_number}!")

        if st.button("Previous Page"):
            st.session_state.Backend_subpage = "main"
        if st.button("Homepage"):
            st.session_state.page = "Home"

    def change_employee_page():
        st.header("Change Employee Data")
        employee_numbers = [
            row[headers.index("EmployeeNumber")] for row in sheet_data[1:]
        ]
        selected_emp = st.selectbox("Select Employee Number", employee_numbers)

        if selected_emp:
            emp_index = employee_numbers.index(selected_emp) + 1
            current_data = sheet_data[emp_index]

            with st.form("change_employee_form"):
                updated_employee = [
                    st.text_input(headers[i], value=current_data[i])
                    for i in range(len(headers))
                ]
                if st.form_submit_button("Update"):
                    worksheet.update(f"A{emp_index + 1}", [updated_employee])
                    st.success("Employee data updated successfully!")

        if st.button("Previous Page"):
            st.session_state.Backend_subpage = "main"
        if st.button("Homepage"):
            st.session_state.page = "Home"


    def delete_employee_page():
        st.header("Delete Employee Data")
        employee_numbers = [
            row[headers.index("EmployeeNumber")] for row in sheet_data[1:]
        ]
        selected_emp = st.selectbox("Select Employee Number to Delete", employee_numbers)

        if selected_emp:
            emp_index = employee_numbers.index(selected_emp) + 1
            if st.button("Delete Employee"):
                worksheet.delete_rows(emp_index + 1)
                st.success(f"Employee {selected_emp} deleted successfully!")

        if st.button("Previous Page"):
            st.session_state.Backend_subpage = "main"
        if st.button("Homepage"):
            st.session_state.page = "Home"


    if "Backend_subpage" not in st.session_state:
        st.session_state.Backend_subpage = "main"

    if st.session_state.Backend_subpage == "main":
        main_page()
    elif st.session_state.Backend_subpage == "add_employee":
        add_employee_page()
    elif st.session_state.Backend_subpage == "change_employee":
        change_employee_page()
    elif st.session_state.Backend_subpage == "delete_employee":
        delete_employee_page()

# Employee Report Page
elif st.session_state.page == "Employee Report":
    st.title("Employee Report")

    # Load OpenAI API Key
    try:
        openai.api_key = st.secrets["openai"]["api_key"]
    except KeyError:
        st.error("OpenAI API Key is missing. Please check your Streamlit secrets.")
        if st.button("Homepage"):
            st.session_state.page = "Home"
        st.stop()

    # Check if EmployeeNumber column is present
    if "EmployeeNumber" not in df.columns:
        st.error("The 'EmployeeNumber' column is missing from the Google Sheet Data.")
        if st.button("Homepage"):
            st.session_state.page = "Home"
        st.stop()

    # Employee selection
    employee_number = st.selectbox("Choose Employee (EmployeeNumber)", df["EmployeeNumber"].unique())

    # Filter data for selected employee
    employee_data = df[df["EmployeeNumber"] == employee_number]
    if employee_data.empty:
        st.error("The selected EmployeeNumber is not present in the data table.")
        if st.button("Homepage"):
            st.session_state.page = "Home"
        st.stop()

    employee_data = employee_data.iloc[0]

    def generate_report(employee):
        # Build the prompt
        prompt = (
            "Create a short, formal, and professional employee report in English using only the provided data. "
            "The employee does not have a name, so please refer to them by their EmployeeNumber. "
            "Do not add any information not present in the data. Present the information as a cohesive paragraph "
            "without additional speculation.\n\n"
            f"Data:\n"
            f"EmployeeNumber: {employee['EmployeeNumber']}\n"
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
