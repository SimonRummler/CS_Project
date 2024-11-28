# Berechnung zusätzlicher Metriken
n = len(y)  # Anzahl der Beobachtungen
k = X.shape[1]  # Anzahl der Prädiktoren

# Adjusted R²
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))

# RMSE
rmse = np.sqrt(mse)

# Modellbewertung anzeigen
st.subheader("Model Metrics")
st.write(f"R²: {r2:.2f}")
st.write(f"Adjusted R²: {adjusted_r2:.2f}")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# **3D-Visualisierung mit zusätzlichen Metriken**
st.subheader("3D Visualization with Model Metrics")
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Datenpunkte mit Farbcodierung nach JobLevel
job_levels = df_filtered["JobLevel"].unique()
colors = sns.color_palette("viridis", len(job_levels))

for i, job_level in enumerate(sorted(job_levels)):
    # Filter für das jeweilige Job Level
    level_data = df_filtered[df_filtered["JobLevel"] == job_level]
    level_predictions = y_pred_full[df_filtered["JobLevel"] == job_level]

    # Punkte über der Regressionslinie markieren
    above_line = level_data["MonthlyIncome"] > level_predictions
    below_line = ~above_line

    # Punkte über der Linie
    ax.scatter(
        level_data["TotalWorkingYears"][above_line],
        level_data["JobLevel"][above_line],
        level_data["MonthlyIncome"][above_line],
        color=colors[i],
        alpha=0.8,
        label=f"Job Level {job_level} (Above Line)",
        marker="^",
    )

    # Punkte unter der Linie
    ax.scatter(
        level_data["TotalWorkingYears"][below_line],
        level_data["JobLevel"][below_line],
        level_data["MonthlyIncome"][below_line],
        color=colors[i],
        alpha=0.5,
        label=f"Job Level {job_level} (Below Line)",
        marker="o",
    )

# Regressionsfläche erstellen
x_surf, y_surf = np.meshgrid(
    np.linspace(df_filtered["TotalWorkingYears"].min(), df_filtered["TotalWorkingYears"].max(), 50),
    np.linspace(df_filtered["JobLevel"].min(), df_filtered["JobLevel"].max(), 50),
)
z_surf = model.predict(scaler.transform(np.c_[x_surf.ravel(), y_surf.ravel()])).reshape(x_surf.shape)

# Regressionsfläche plotten
ax.plot_surface(x_surf, y_surf, z_surf, cmap="viridis", alpha=0.3, edgecolor="none")

# Achsentitel und Beschriftungen
ax.set_xlabel("Total Working Years")
ax.set_ylabel("Job Level")
ax.set_zlabel("Monthly Income")
ax.set_title("3D Regression with Metrics")

# **Anpassung der Perspektive**
ax.view_init(elev=15, azim=-30)  # Perspektive anpassen

# Metriken als Text anzeigen
ax.text2D(0.05, 0.95, f"R²: {r2:.2f}", transform=ax.transAxes, fontsize=10, color="black")
ax.text2D(0.05, 0.90, f"Adjusted R²: {adjusted_r2:.2f}", transform=ax.transAxes, fontsize=10, color="black")
ax.text2D(0.05, 0.85, f"RMSE: {rmse:.2f}", transform=ax.transAxes, fontsize=10, color="black")
ax.text2D(0.05, 0.80, f"MSE: {mse:.2f}", transform=ax.transAxes, fontsize=10, color="black")

# Legende hinzufügen
ax.legend(loc="best")
st.pyplot(fig)


