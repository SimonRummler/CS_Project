import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Titel der App
st.title("Multiple Regression: Monthly Income Prediction")

# CSV-Datei hochladen
uploaded_file = st.file_uploader("HR_Dataset_Group4.5.csv", type=["csv"])

if uploaded_file:
    # Daten laden
    df = pd.read_csv(uploaded_file, sep=';')  # Anpassung des Trennzeichens, falls notwendig
    
    # Sicherstellen, dass die relevanten Spalten vorhanden sind
    try:
        df = df[['TotalWorkingYears', 'JobLevel', 'MonthlyIncome']].dropna()

        # Features und Ziel definieren
        X = df[['TotalWorkingYears', 'JobLevel']]
        y = df['MonthlyIncome']

        # Datenaufteilung in Trainings- und Testdaten
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Modelltraining
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Vorhersagen und Metriken
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Regressionsergebnisse anzeigen
        st.subheader("Regression Results")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R² Score: {r2:.2f}")
        st.write(f"Model Coefficients: {model.coef_}")
        st.write(f"Intercept: {model.intercept_}")

        # Visualisierung
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
        ax.set_xlabel("Actual Monthly Income")
        ax.set_ylabel("Predicted Monthly Income")
        ax.set_title("Actual vs Predicted Monthly Income")
        st.pyplot(fig)

    except KeyError as e:
        st.error(f"Missing column: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

