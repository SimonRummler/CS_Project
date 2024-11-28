import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Titel der Streamlit-App
st.title("Logistic Regression: Distance from Home vs. Attrition")

# Daten aus der CSV-Datei laden
csv_file = "HR_Dataset_Group4.5.csv"  # Dateiname im Repository
try:
    df = pd.read_csv(csv_file, sep=";")  # Semikolon als Trennzeichen
except FileNotFoundError:
    st.error("CSV file not found. Please check the file path.")
    st.stop()

# Prüfen, ob die erforderlichen Spalten existieren
if "DistanceFromHome" in df.columns and "Attrition" in df.columns:
    # Relevante Spalten filtern und NaN-Werte entfernen
    df_filtered = df[["DistanceFromHome", "Attrition"]].dropna()
    df_filtered["Attrition"] = df_filtered["Attrition"].map({"Yes": 1, "No": 0})

    # Features (X) und Zielvariable (y) definieren
    X = df_filtered[["DistanceFromHome"]]
    y = df_filtered["Attrition"]

    # Standardisierung der Distanzdaten
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-Test-Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Modell trainieren
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Vorhersagen für Testdaten
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Ergebnisse anzeigen
    st.subheader("Model Evaluation")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Confusion Matrix:")
    st.write(cm)
    st.write("Classification Report:")
    st.write(pd.DataFrame(report).transpose())

    # Wahrscheinlichkeiten berechnen
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_range_scaled = scaler.transform(X_range)
    y_prob = model.predict_proba(X_range_scaled)[:, 1]

    # Benutzer wählt Visualisierung aus
    st.sidebar.header("Select Visualization")
    viz_option = st.sidebar.selectbox(
        "Choose the visualization type:",
        options=["Scatterplot with Regression Curve", "Density Plot (KDE)", "Grouped Bar Chart"]
    )

    # Visualisierung 1: Scatterplot mit Regressionskurve
    if viz_option == "Scatterplot with Regression Curve":
        st.subheader("Scatterplot with Logistic Regression Curve")
        fig, ax = plt.subplots(figsize=(10, 6))
        class_0 = df_filtered[df_filtered["Attrition"] == 0]
        class_1 = df_filtered[df_filtered["Attrition"] == 1]
        ax.scatter(class_0["DistanceFromHome"], class_0["Attrition"], color="blue", alpha=0.6, label="Attrition = No")
        ax.scatter(class_1["DistanceFromHome"], class_1["Attrition"], color="orange", alpha=0.6, label="Attrition = Yes")
        ax.plot(X_range, y_prob, color="red", linewidth=2, label="Logistic Regression Curve")
        ax.axhline(0.5, color="green", linestyle="--", label="50% Probability Threshold")
        ax.set_xlabel("Distance from Home (km)", fontsize=12)
        ax.set_ylabel("Probability of Attrition", fontsize=12)
        ax.set_title("Distance from Home vs. Attrition", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)
        st.pyplot(fig)

    # Visualisierung 2: Density Plot (KDE)
    elif viz_option == "Density Plot (KDE)":
        st.subheader("Density Plot (KDE) with Logistic Regression Curve")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(df_filtered[df_filtered["Attrition"] == 0]["DistanceFromHome"], 
                    label="Attrition = No", fill=True, alpha=0.6, color="blue", ax=ax)
        sns.kdeplot(df_filtered[df_filtered["Attrition"] == 1]["DistanceFromHome"], 
                    label="Attrition = Yes", fill=True, alpha=0.6, color="orange", ax=ax)
        ax.plot(X_range, y_prob, color="red", linewidth=2, label="Logistic Regression Curve")
        ax.set_xlabel("Distance from Home (km)", fontsize=12)
        ax.set_ylabel("Density / Probability", fontsize=12)
        ax.set_title("KDE Plot with Logistic Regression", fontsize=14)
        ax.legend(fontsize=10)
        st.pyplot(fig)

    # Visualisierung 3: Gruppierte Balkendiagramme
    elif viz_option == "Grouped Bar Chart":
        st.subheader("Grouped Bar Chart")
        df_filtered["DistanceGroup"] = pd.cut(df_filtered["DistanceFromHome"], bins=5)
        grouped = df_filtered.groupby(["DistanceGroup", "Attrition"]).size().unstack()
        fig, ax = plt.subplots(figsize=(10, 6))
        grouped.plot(kind="bar", stacked=True, color=["blue", "orange"], ax=ax)
        ax.set_xlabel("Distance from Home Groups", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Attrition Counts by Distance from Home Groups", fontsize=14)
        ax.legend(["Attrition = No", "Attrition = Yes"], fontsize=10)
        st.pyplot(fig)

else:
    st.error("The required columns 'DistanceFromHome' and 'Attrition' are not found in the dataset.")

