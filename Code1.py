import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")

# Load the uploaded CSV file
df = pd.read_csv(r"CS_Project/HR_Dataset_Group4.5.csv", delimiter=';')

# Display dataframe in Streamlit
st.write("## HR Dataset")
st.dataframe(df)

# Connect to SQLite and create new database
conn = sqlite3.connect("hr_full_database_group4_5.db")
cursor = conn.cursor()

# Create table if not exists using all columns from the CSV
sql_create_table = """
CREATE TABLE IF NOT EXISTS hr_full_data (
    Age INTEGER,
    Attrition TEXT,
    BusinessTravel TEXT,
    DailyRate INTEGER,
    Department TEXT,
    DistanceFromHome INTEGER,
    Education INTEGER,
    EducationField TEXT,
    EmployeeCount INTEGER,
    EmployeeNumber INTEGER PRIMARY KEY,
    EnvironmentSatisfaction INTEGER,
    Gender TEXT,
    HourlyRate INTEGER,
    JobInvolvement INTEGER,
    JobLevel INTEGER,
    JobRole TEXT,
    JobSatisfaction INTEGER,
    MaritalStatus TEXT,
    MonthlyIncome INTEGER,
    MonthlyRate INTEGER,
    NumCompaniesWorked INTEGER,
    Over18 TEXT,
    OverTime TEXT,
    PercentSalaryHike INTEGER,
    PerformanceRating INTEGER,
    RelationshipSatisfaction INTEGER,
    StandardHours INTEGER,
    StockOptionLevel INTEGER,
    TotalWorkingYears INTEGER,
    TrainingTimesLastYear INTEGER,
    WorkLifeBalance INTEGER,
    YearsAtCompany INTEGER,
    YearsInCurrentRole INTEGER,
    YearsSinceLastPromotion INTEGER,
    YearsWithCurrManager INTEGER
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
)
VALUES (
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

# Commit changes and close connection
conn.commit()
conn.close()

# Selects only the input department rows
def return_row(department):
    conn = sqlite3.connect("hr_full_database_group4_5.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM hr_full_data WHERE Department = ?", (department,))
    rows = cursor.fetchall()
    conn.close()
    return rows

# Funktion Definition print data
def print_department_data(department):
    conn = sqlite3.connect("hr_full_database_group4_5.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM hr_full_data WHERE Department = ?", (department,))
    rows = cursor.fetchall()

    column_names = [description[0] for description in cursor.description]
    conn.close()
    st.write(f"### Daten für Abteilung: {department}")
    for row in rows:
        row_dict = dict(zip(column_names, row))
        st.write(row_dict)

# Funktion für Prozentanteil weiblicher Mitarbeiter
def percentage_female(department):
    rows = return_row(department)
    length_employe = len(rows)
    sum = 0
    for row in rows:
        if row[11] == "Female":
            sum += 1
    return (sum / length_employe)

# Funktion für Visualisierung der weiblichen Mitarbeiter
def visualize_female_data(percentage_female_num, department):
    labels = ['Female', 'Male']
    percentage_male_num = 1 - percentage_female_num
    colors = ['#ff9999', '#66b3ff']
    sizes = [percentage_female_num, percentage_male_num]
    explode = (0.1, 0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=140)
    ax.set_title('Percentage of women in ' + department)
    st.pyplot(fig)

# Funktion für Studienfelder
def fields_of_study(department):
    rows = return_row(department)
    length_employe = len(rows)

    human_resources = 0
    life_sciences = 0
    marketing = 0
    medical = 0
    other = 0
    technical_degree = 0

    for row in rows:
        if row[7] == "Life Sciences":
            life_sciences += 1
        elif row[7] == "Human Resources":
            human_resources += 1
        elif row[7] == "Marketing":
            marketing += 1
        elif row[7] == "Medical":
            medical += 1
        elif row[7] == "Technical Degree":
            technical_degree += 1
        elif row[7] == "Other":
            other += 1

    human_resources /= length_employe
    life_sciences /= length_employe
    marketing /= length_employe
    medical /= length_employe
    other /= length_employe
    technical_degree /= length_employe

    return (human_resources, life_sciences, marketing, medical, other, technical_degree)

# Funktion für Visualisierung der Studienfelder
def visualize_fields_of_study(result_fields_of_study, department):
    labels = ['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree']
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FFFF33', '#FF33FF', '#33FFFF']
    sizes = list(result_fields_of_study)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=140)
    ax.set_title('Fields of study in ' + department + ':')
    st.pyplot(fig)

# Benutzerabfrage und Funktionsaufrufe in Streamlit
department = st.text_input("Bitte geben Sie ihr Department ein:")

if department:
    # Funktion Aufruf print data
    print_department_data(department)

    # Funktion Aufruf female und print
    st.write(f"### This is the female percentage of {department}:")
    result_female = percentage_female(department)
    st.write(result_female)

    # Funktion Aufruf visualize female data
    visualize_female_data(result_female, department)

    # Funktion Aufruf field of study und print
    st.write(f"### This is the field of study of {department}:")
    result_fields_of_study = fields_of_study(department)
    st.write(result_fields_of_study)

    # Funktion Aufruf visualize field of study
    visualize_fields_of_study(result_fields_of_study, department)

