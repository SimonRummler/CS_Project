import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
import seaborn as sns
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

# Lineare Regression: Total Working Years vs. Monthly Income
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

# Logistische Regression: Distance from Home vs. Attrition
st.header("Logistic Regression: Distance from Home vs. Attrition")
if "DistanceFromHome" in df.columns and "Attrition" in df.columns:
    df_filtered = df[["DistanceFromHome", "Attrition"]].dropna()
    df_filtered["Attrition"] = df_filtered["Attrition"].map({"Yes": 1, "No": 0})

    X = df_filtered[["DistanceFromHome"]]
    y = df_filtered["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x="DistanceFromHome", y="Attrition", data=df_filtered, alpha=0.7, ax=ax)
    ax.set_xlabel("Distance from Home (km)")
    ax.set_ylabel("Attrition (1 = Yes, 0 = No)")
    ax.set_title("Logistic Regression: Distance from Home vs. Attrition")
    
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_prob = model.predict_proba(X_range)[:, 1]
    ax.plot(X_range, y_prob, color="red", linewidth=2, label="Logistic Regression Curve")
    ax.legend()
    st.pyplot(fig)
else:
    st.error("The required columns 'DistanceFromHome' and 'Attrition' are not found in the dataset.")
