import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

# Titel der Streamlit-App
st.title("HR Analytics: Exploring Relationships Between Variables")

# Daten aus der CSV-Datei laden
csv_file = "HR_Dataset_Group4.5.csv"  # Dateiname im Repository
try:
    df = pd.read_csv(csv_file, sep=";")  # Semikolon als Trennzeichen
except FileNotFoundError:
    st.error("CSV file not found. Please check the file path.")
    st.stop()

# Abschnitt 1: Lineare Regression: Total Working Years vs. Monthly Income
st.header("Linear Regression: Total Working Years vs. Monthly Income")
if "TotalWorkingYears" in df.columns and "MonthlyIncome" in df.columns:
    df_filtered = df[["TotalWorkingYears", "MonthlyIncome"]].dropna()

    # Entferne Ausreißer mit IQR
    Q1 = df_filtered["MonthlyIncome"].quantile(0.25)
    Q3 = df_filtered["MonthlyIncome"].quantile(0.75)
    IQR = Q3 - Q1
    df_filtered = df_filtered[
        (df_filtered["MonthlyIncome"] >= Q1 - 1.5 * IQR) & 
        (df_filtered["MonthlyIncome"] <= Q3 + 1.5 * IQR)
    ]

    X = df_filtered[["TotalWorkingYears"]]
    y = df_filtered["MonthlyIncome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Evaluation")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', s=10, alpha=0.7, label='Data Points')
    ax.plot(X, model.predict(X), color='red', linewidth=2, label='Linear Regression')
    ax.set_xlabel("Total Working Years (Years)", fontsize=12)
    ax.set_ylabel("Monthly Income (in USD)", fontsize=12)
    ax.set_title("Linear Regression: Total Working Years vs. Monthly Income", fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
else:
    st.error("The required columns 'TotalWorkingYears' and 'MonthlyIncome' are not found in the dataset.")

# Abschnitt 2: Logistische Regression: Distance from Home vs. Attrition
st.header("Logistic Regression: Distance from Home vs. Attrition")
if "DistanceFromHome" in df.columns and "Attrition" in df.columns:
    df_filtered = df[["DistanceFromHome", "Attrition"]].dropna()
    df_filtered["Attrition"] = df_filtered["Attrition"].map({"Yes": 1, "No": 0})

    X = df_filtered[["DistanceFromHome"]]
    y = df_filtered["Attrition"]

    # Standardisierung der Distanzdaten
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.subheader("Model Evaluation")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Confusion Matrix:")
    st.write(cm)
    st.write("Classification Report:")
    st.write(pd.DataFrame(report).transpose())

    # Visualisierung nach Klassen
    fig, ax = plt.subplots(figsize=(10, 6))
    class_0 = df_filtered[df_filtered["Attrition"] == 0]
    class_1 = df_filtered[df_filtered["Attrition"] == 1]

    ax.scatter(class_0["DistanceFromHome"], class_0["Attrition"], color='blue', s=20, alpha=0.7, label='Attrition = No')
    ax.scatter(class_1["DistanceFromHome"], class_1["Attrition"], color='orange', s=20, alpha=0.7, label='Attrition = Yes')

    # Logistische Regressionskurve hinzufügen
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_range_scaled = scaler.transform(X_range)  # Skaliere den Bereich
    y_prob = model.predict_proba(X_range_scaled)[:, 1]
    ax.plot(X_range, y_prob, color="red", linewidth=2, label="Logistic Regression Curve")

    ax.set_xlabel("Distance from Home (km)", fontsize=12)
    ax.set_ylabel("Probability of Attrition", fontsize=12)
    ax.set_title("Logistic Regression: Distance from Home vs. Attrition", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.7)
    st.pyplot(fig)
else:
    st.error("The required columns 'DistanceFromHome' and 'Attrition' are not found in the dataset.")


