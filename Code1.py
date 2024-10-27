import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from PIL import Image

# Streamlit Title and Description
st.title("HR Data Analysis Tool")
st.write("Upload an HR dataset CSV file to analyze data by department.")

# File Uploader for CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Load CSV file
    df = pd.read_csv(uploaded_file, delimiter=';')

    # Connect to SQLite and create new database
    conn = sqlite3.connect("hr_full_database_group4_5.db")
    cursor = conn.cursor()

    # Create table if not exists using all columns from the CSV
    sql_create_table = """
    CREATE TABLE IF NOT EXISTS hr_full_data (
        Age INTEGER, Attrition TEXT, BusinessTravel TEXT, DailyRate INTEGER,
        Department TEXT, DistanceFromHome INTEGER, Education INTEGER, EducationField TEXT,
        EmployeeCount INTEGER, EmployeeNumber INTEGER PRIMARY KEY, EnvironmentSatisfaction INTEGER,
        Gender TEXT, HourlyRate INTEGER, JobInvolvement INTEGER, JobLevel INTEGER, JobRole TEXT,
        JobSatisfaction INTEGER, MaritalStatus TEXT, MonthlyIncome INTEGER, MonthlyRate INTEGER,
        NumCompaniesWorked INTEGER, Over18 TEXT, OverTime TEXT, PercentSalaryHike INTEGER,
        PerformanceRating INTEGER, RelationshipSatisfaction INTEGER, StandardHours INTEGER,
        StockOptionLevel INTEGER, TotalWorkingYears INTEGER, TrainingTimesLastYear INTEGER,
        WorkLifeBalance INTEGER, YearsAtCompany INTEGER, YearsInCurrentRole INTEGER,
        YearsSinceLastPromotion INTEGER, YearsWithCurrManager INTEGER
    );
    """
    cursor.execute(sql_create_table)

    # Insert all data from dataframe into the table
    df_records = df.to_dict(orient='records')
    sql_insert = """
    INSERT OR REPLACE INTO hr_full_data (
        Age, Attrition, BusinessTravel, DailyRate, Department, DistanceFromHome,
        Education, EducationField, EmployeeCount, EmployeeNumber, EnvironmentSatisfaction,
        Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction,
        MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked, Over18, OverTime,
        PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StandardHours,
        StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance,
        YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager
    ) VALUES (
        :Age, :Attrition, :BusinessTravel, :DailyRate, :Department, :DistanceFromHome,
        :Education, :EducationField, :EmployeeCount, :EmployeeNumber, :EnvironmentSatisfaction,
        :Gender, :HourlyRate, :JobInvolvement, :JobLevel, :JobRole, :JobSatisfaction,
        :MaritalStatus, :MonthlyIncome, :MonthlyRate, :NumCompaniesWorked, :Over18, :OverTime,
        :PercentSalaryHike, :PerformanceRating, :RelationshipSatisfaction, :StandardHours,
        :StockOptionLevel, :TotalWorkingYears, :TrainingTimesLastYear, :WorkLifeBalance,
        :YearsAtCompany, :YearsInCurrentRole, :YearsSinceLastPromotion, :YearsWithCurrManager
    );
    """
    cursor.executemany(sql_insert, df_records)
    conn.commit()

    # Functions for data processing
    def return_row(department):
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM hr_full_data WHERE Department = ?", (department,))
        rows = cursor.fetchall()
        return rows

    def percentage_female(department):
        rows = return_row(department)
        total = len(rows)
        females = sum(1 for row in rows if row[11] == "Female")
        return females / total if total > 0 else 0

    def visualize_female_data(percentage_female_num, department):
        labels = ['Female', 'Male']
        sizes = [percentage_female_num, 1 - percentage_female_num]
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, explode=(0.1, 0))
        ax.set_title(f'Female Percentage in {department}')
        st.pyplot(fig)

    def fields_of_study(department):
        rows = return_row(department)
        field_counts = pd.Series(row[7] for row in rows).value_counts(normalize=True)
        return field_counts

    def visualize_fields_of_study(field_counts, department):
        fig, ax = plt.subplots()
        ax.pie(field_counts, labels=field_counts.index, autopct='%1.1f%%', startangle=140)
        ax.set_title(f'Fields of Study in {department}')
        st.pyplot(fig)

    # Input Department and Visualizations
    department = st.text_input("Enter Department for Analysis:")
    if department:
        st.write(f"Data for Department: {department}")
        
        # Female Percentage
        result_female = percentage_female(department)
        st.write(f"Female Percentage in {department}: {result_female:.2%}")
        visualize_female_data(result_female, department)

        # Fields of Study
        result_fields_of_study = fields_of_study(department)
        st.write(f"Fields of Study Distribution in {department}")
        st.write(result_fields_of_study)
        visualize_fields_of_study(result_fields_of_study, department)
    
    # Close database connection after all operations
    conn.close()
