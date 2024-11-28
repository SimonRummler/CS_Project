import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Titel der Streamlit-App
st.title("Interactive Linear Regression: Select Inputs and Outputs")

# Daten aus der CSV-Datei laden
csv_file = "HR_Dataset_Group4.5.csv"  # Dateiname im Repository
try:
    df = pd.read_csv(csv_file, sep=";")  # Semikolon als Trennzeichen
except FileNotFoundError:
    st.error("CSV file not found. Please check the file path.")
    st.stop()

# Liste der verfügbaren Spalten
columns = df.columns.tolist()

# Benutzer wählt Eingabe- und Zielvariablen aus
st.sidebar.header("Select Variables")
input_features = st.sidebar.multiselect("Select Input Features", options=columns, default=["TotalWorkingYears", "JobLevel"])
target_variable = st.sidebar.selectbox("Select Target Variable", options=columns, index=columns.index("MonthlyIncome"))

# Prüfen, ob die ausgewählten Variablen in den Daten existieren
if set(input_features).issubset(columns) and target_variable in columns:
    # Relevante Spalten filtern und NaN-Werte entfernen
    selected_columns = input_features + [target_variable]
    df_filtered = df[selected_columns].dropna()

    # Entferne Ausreißer mit IQR für die Zielvariable
    Q1 = df_filtered[target_variable].quantile(0.25)
    Q3 = df_filtered[target_variable].quantile(0.75)
    IQR = Q3 - Q1
    df_filtered = df_filtered[
        (df_filtered[target_variable] >= Q1 - 1.5 * IQR) & 
        (df_filtered[target_variable] <= Q3 + 1.5 * IQR)
    ]

    # Features (X) und Zielvariable (y) definieren
    X = df_filtered[input_features]
    y = df_filtered[target_variable]

    # Daten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modell trainieren
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Vorhersagen für Testdaten
    y_pred = model.predict(X_test)

    # Modellbewertung
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Ergebnisse anzeigen
    st.subheader("Model Evaluation")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    # Interpretation des MSE
    st.subheader("Interpretation of Results")
    mean_target = y.mean()
    st.write(f"Mean of Target Variable ({target_variable}): {mean_target:.2f}")
    st.write("Interpretation of MSE:")
    if mse < 0.1 * mean_target:
        st.success("The MSE is very low relative to the target variable's mean. This indicates a very good model fit.")
    elif mse < 0.3 * mean_target:
        st.info("The MSE is moderate. The model explains some variability but could be improved.")
    else:
        st.warning("The MSE is high relative to the target variable's mean. The model may not be reliable.")

    # Visualisierung der Daten
    st.subheader("Visualization")
    if len(input_features) == 1:  # Scatterplot nur für eine Eingabevariable
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X, y, color='blue', s=20, alpha=0.7, label="Data Points")
        ax.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
        ax.set_xlabel(input_features[0])
        ax.set_ylabel(target_variable)
        ax.set_title(f"{input_features[0]} vs. {target_variable}")
        ax.legend()
        st.pyplot(fig)
    else:  # Keine Visualisierung bei mehreren Eingabevariablen
        st.write("Visualization is not supported for multiple input features.")

else:
    st.error("The selected input features or target variable are not available in the dataset.")
