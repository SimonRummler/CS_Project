# **3D-Visualisierung mit angepasster Perspektive**
st.subheader("3D Visualization with Adjusted Perspective and Job Level Coloring")
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
ax.set_title("3D Regression with Job Levels and Deviations")

# **Anpassung der Perspektive**
ax.view_init(elev=10, azim=-60)  # Elevation und Azimut angepasst für die gewünschte Ansicht

# Legende hinzufügen
ax.legend(loc="best")
st.pyplot(fig)
