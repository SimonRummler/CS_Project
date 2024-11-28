# Eingabefelder für Benutzer
st.subheader("Predict Monthly Income and Job Level")
total_working_years = st.number_input("Enter Total Working Years", min_value=0, max_value=50, step=1)

# Initialisierung der Variablen
predicted_income = None
predicted_joblevel = None

if st.button("Predict"):
    input_data = np.array([[total_working_years]])
    input_scaled = scaler.transform(input_data)

    # Vorhersage
    predicted_income = income_model.predict(input_scaled)[0]
    predicted_joblevel = joblevel_model.predict(input_scaled)[0]

    st.write(f"Predicted Monthly Income: ${predicted_income:.2f}")
    st.write(f"Predicted Job Level (rounded): {round(predicted_joblevel)}")

# Hauptplot mit Konfidenzintervallen
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

# Nur anzeigen, wenn eine Vorhersage existiert
if predicted_income is not None:
    ax.scatter(total_working_years, predicted_income, color="black", s=100, label="Predicted Income", zorder=5)

# Achsentitel und Beschriftungen
ax.set_xlabel("Total Working Years (Years)", fontsize=12)
ax.set_ylabel("Monthly Income (in USD)", fontsize=12)
ax.set_title("Linear Regression: Total Working Years vs. Monthly Income", fontsize=14)
ax.legend()
st.pyplot(fig)

