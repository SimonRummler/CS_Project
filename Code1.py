import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")
from PIL import Image
from google.oauth2 import service_account
from googleapiclient.discovery import build

# 1. Lade die JSON-Datei mit den Anmeldedaten
SERVICE_ACCOUNT_FILE = r"CS_Project/round-seeker-439709-p8-60582c389ba4.json"
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# Erstelle die Anmeldedaten über die JSON-Datei
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# 2. Google Sheets API initialisieren
service = build('sheets', 'v4', credentials=creds)

# 3. Informationen zur Tabelle
SAMPLE_SPREADSHEET_ID = '19CC438qwcEpCufbyukbzQ1RmVW9uZ1VK6rFHtXNU8IU'  # Google Spreadsheet ID hier einfügen
SHEET_NAME = "HR-Data"  # Tabellenblatt Name hier einfügen
SAMPLE_RANGE_NAME = f'{SHEET_NAME}!A1:AI1471' # Bereich zum Lesen der Daten

# 4. Daten aus der Google-Tabelle lesen
sheet = service.spreadsheets()
result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID, range=SAMPLE_RANGE_NAME).execute()
values = result.get('values', [])

# Umwandeln der Tabellen-Daten in ein pandas DataFrame
if not values:
    print('No Data found')
    exit()
else:
    column_names = values[0]  # Erste Zeile enthält die Spaltennamen
    data = values[1:]  # Die restlichen Zeilen enthalten die Daten
    df = pd.DataFrame(data, columns=column_names)

# Konvertiere relevante Spalten zu numerischen Werten
numerical_columns = ["Age", "DailyRate", "DistanceFromHome", "MonthlyIncome", "NumCompaniesWorked", "TotalWorkingYears", "YearsAtCompany"]
for column in numerical_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Funktion zur Rückgabe von Reihen eines bestimmten Departments
def return_row(department):
    return df[df['Department'] == department]

# Funktion zur Ausgabe der Daten eines bestimmten Departments
def print_department_data(department):
    rows = return_row(department)
    print(rows)

# Funktion zur Berechnung des Anteils weiblicher Mitarbeiter
def percentage_female(department):
    rows = return_row(department)
    total_employees = len(rows)
    if total_employees == 0:
        return 0
    female_count = len(rows[rows['Gender'] == 'Female'])
    return female_count / total_employees

# Funktion zur Visualisierung des Anteils weiblicher Mitarbeiter
def visualize_female_data(percentage_female_num, department):
    labels = ['Female', 'Male']
    percentage_male_num = 1 - percentage_female_num
    colors = ['#ff9999', '#66b3ff']
    sizes = [percentage_female_num, percentage_male_num]
    explode = (0.1, 0)

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140)
    plt.title('Percentage of women in ' + department)
    plt.savefig(f'{department}_female_percentage.png')
    plt.close()

# Funktion zur Rückgabe des Bildungsbereichs
def fields_of_study(department):
    rows = return_row(department)
    length_employe = len(rows)

    field_counts = {
        "Human Resources": 0,
        "Life Sciences": 0,
        "Marketing": 0,
        "Medical": 0,
        "Technical Degree": 0,
        "Other": 0
    }

    for field in rows['EducationField']:
        if field in field_counts:
            field_counts[field] += 1

    # Umwandlung in Prozentsätze
    for field in field_counts:
        field_counts[field] /= length_employe if length_employe > 0 else 1

    return field_counts.values()

# Funktion zur Visualisierung der Bildungsbereiche
def visualize_fields_of_study(result_fields_of_study, department):
    labels = ['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree']
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FFFF33', '#FF33FF', '#33FFFF']
    sizes = list(result_fields_of_study)

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title('Fields of study in ' + department + ':')
    plt.savefig(f'{department}_fields_of_study.png')
    plt.close()

# Hauptprogramm
if __name__ == "__main__":
    department = input("Please Enter the Department: ")

    # Ausgabe der Daten des Departments
    print_department_data(department)

    # Berechnung und Ausgabe des Anteils weiblicher Mitarbeiter
    print(f"This is the female percentage of {department}:")
    result_female = percentage_female(department)
    print(result_female)

    # Visualisierung des Anteils weiblicher Mitarbeiter
    visualize_female_data(result_female, department)
    image_path1 = f"{department}_female_percentage.png"
    img1 = Image.open(image_path1)
    img1.show()

    # Ausgabe des Bildungsbereichs und Visualisierung
    print(f"This is the field of study of {department}:")
    result_fields_of_study = fields_of_study(department)
    print(result_fields_of_study)

    visualize_fields_of_study(result_fields_of_study, department)
    image_path2 = f"{department}_fields_of_study.png"
    img2 = Image.open(image_path2)
    img2.show()
