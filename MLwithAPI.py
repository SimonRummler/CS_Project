# Titel der Streamlit-App
st.title("Linear Regression: Total Working Years vs. Monthly Income (JobLevel for Visualization)")

# Daten aus der CSV-Datei laden
csv_file = "HR_Dataset_Group4.5.csv"  # Dateiname im Repository
try:
    df = pd.read_csv(csv_file, sep=";")  # Semikolon als Trennzeichen
except FileNotFoundError:
    st.error("CSV file not found. Please check the file path.")
    st.stop()

# Prüfen, ob die erforderlichen Spalten existieren
if "TotalWorkingYears" in df.columns and "MonthlyIncome" in df.columns and "JobLevel" in df.columns:
    # Relevante Spalten filtern und NaN-Werte entfernen
    df_filtered = df[["TotalWorkingYears", "MonthlyIncome", "JobLevel"]].dropna()

    # Features und Zielvariable definieren
    X = df_filtered[["TotalWorkingYears"]].values  # Nur TotalWorkingYears als Input
    y_income = df_filtered["MonthlyIncome"].values
    y_joblevel = df_filtered["JobLevel"].values

    # Standardisierung der Eingabedaten
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-Test-Split für beide Modelle
    X_train, X_test, y_income_train, y_income_test = train_test_split(X_scaled, y_income, test_size=0.2, random_state=42)
    _, _, y_joblevel_train, y_joblevel_test = train_test_split(X_scaled, y_joblevel, test_size=0.2, random_state=42)

    # Modelle trainieren
    income_model = LinearRegression()
    income_model.fit(X_train, y_income_train)

    joblevel_model = LinearRegression()
    joblevel_model.fit(X_train, y_joblevel_train)

    # Vorhersagen für Testdaten
    y_income_pred = income_model.predict(X_test)
    y_joblevel_pred = joblevel_model.predict(X_test)

    # Modellbewertung
    mse_income = mean_squared_error(y_income_test, y_income_pred)
    r2_income = r2_score(y_income_test, y_income_pred)
    mse_joblevel = mean_squared_error(y_joblevel_test, y_joblevel_pred)
    r2_joblevel = r2_score(y_joblevel_test, y_joblevel_pred)

    # Ergebnisse anzeigen
    st.subheader("Model Evaluation")
    st.write(f"Income Model - Mean Squared Error (MSE): {mse_income:.2f}, R² Score: {r2_income:.2f}")
    st.write(f"Job Level Model - Mean Squared Error (MSE): {mse_joblevel:.2f}, R² Score: {r2_joblevel:.2f}")

    # Eingabefelder für Benutzer
    st.subheader("Predict Monthly Income and Job Level")
    total_working_years = st.number_input("Enter Total Working Years", min_value=0, max_value=50, step=1)

    predicted_income = None
    predicted_joblevel = None

    if st.button("Predict"):
        # Neue Eingaben standardisieren
        input_data = np.array([[total_working_years]])
        input_scaled = scaler.transform(input_data)

        # Vorhersage basierend auf Eingaben
        predicted_income = income_model.predict(input_scaled)[0]
        predicted_joblevel = joblevel_model.predict(input_scaled)[0]

        st.write(f"Predicted Monthly Income: ${predicted_income:.2f}")
        st.write(f"Predicted Job Level (rounded): {round(predicted_joblevel)}")

    # Konfidenzintervalle berechnen
    preds_income = []
    preds_joblevel = []
    for _ in range(100):  # Bootstrap-Resampling
        X_sample, y_income_sample = resample(X_train, y_income_train)
        y_joblevel_sample = resample(y_joblevel_train, random_state=42)[0]
        income_model.fit(X_sample, y_income_sample)
        joblevel_model.fit(X_sample, y_joblevel_sample)
        preds_income.append(income_model.predict(X_scaled))
        preds_joblevel.append(joblevel_model.predict(X_scaled))
    lower_income = np.percentile(preds_income, 2.5, axis=0)
    upper_income = np.percentile(preds_income, 97.5, axis=0)
    lower_joblevel = np.percentile(preds_joblevel, 2.5, axis=0)
    upper_joblevel = np.percentile(preds_joblevel, 97.5, axis=0)

    # Hauptplot mit Konfidenzintervallen und Farbkodierung nach JobLevel
    st.subheader("Visualization: Regression with Confidence Intervals")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        X.flatten(), y_income, c=df_filtered["JobLevel"], cmap="viridis", s=10, alpha=0.5, label="Data Points"
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Job Level")

    # Konfidenzintervall hinzufügen
    ax.fill_between(
        X.flatten(), lower_income, upper_income, color="gray", alpha=0.3, label="Confidence Interval"
    )

    # Regressionslinie
    ax.plot(X.flatten(), income_model.predict(X_scaled), color="red", linewidth=2, label="Income Regression Line")

    # Wenn Vorhersage gemacht wurde, füge schwarzen Punkt hinzu
    if predicted_income is not None:
        ax.scatter(total_working_years, predicted_income, color="black", s=100, label="Predicted Income", zorder=5)

    # Achsentitel und Beschriftungen
    ax.set_xlabel("Total Working Years (Years)", fontsize=12)
    ax.set_ylabel("Monthly Income (in USD)", fontsize=12)
    ax.set_title("Linear Regression: Total Working Years vs. Monthly Income", fontsize=14)
    ax.legend()
    st.pyplot(fig)

else:
    st.error("The required columns 'TotalWorkingYears', 'MonthlyIncome', and 'JobLevel' are not found in the dataset.")

