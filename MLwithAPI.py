# ... (Ihr bisheriger Code)

# 3D-Visualisierung mit 30-Grad-Drehung
st.subheader("3D Visualization with 30-Degree Rotation to the Left")
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Datenpunkte mit Farbcodierung nach JobLevel
job_levels = df_filtered["JobLevel"].unique()
colors = sns.color_palette("viridis", len(job_levels))

# Listen für Legenden-Handles und -Labels
handles = []
labels = []

for i, job_level in enumerate(sorted(job_levels)):
    # Filter für das jeweilige Job Level
    level_data = df_filtered[df_filtered["JobLevel"] == job_level]
    if level_data.empty:
        st.warning(f"No data found for Job Level {job_level}.")
        continue

    level_predictions = y_pred_full[df_filtered["JobLevel"] == job_level]

    # Punkte über der Regressionslinie markieren
    above_line = level_data["MonthlyIncome"] > level_predictions
    below_line = ~above_line

    # Punkte über der Linie
    sc_above = ax.scatter(
        level_data["TotalWorkingYears"][above_line],
        level_data["JobLevel"][above_line],
        level_data["MonthlyIncome"][above_line],
        color=colors[i],
        alpha=0.8,
        marker="^",
    )
    # Legenden-Handles und -Labels sammeln
    handles.append(sc_above)
    labels.append(f"Job Level {job_level} (Above Line)")

    # Punkte unter der Linie
    sc_below = ax.scatter(
        level_data["TotalWorkingYears"][below_line],
        level_data["JobLevel"][below_line],
        level_data["MonthlyIncome"][below_line],
        color=colors[i],
        alpha=0.5,
        marker="o",
    )
    # Legenden-Handles und -Labels sammeln
    handles.append(sc_below)
    labels.append(f"Job Level {job_level} (Below Line)")

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
ax.set_title("3D Regression with 30-Degree Rotation to the Left")

# Anpassung der Perspektive
ax.view_init(elev=10, azim=-38)

# Legende hinzufügen
ax.legend(handles, labels, loc="best")
st.pyplot(fig)

