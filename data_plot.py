import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === SETTINGS ===
#log_path = "data/flat_add_mass_log.csv"
#log_path = "data/slope_add_noise_and_mass_log.csv"
log_path = "data/random_terrain_log.csv"


# vel = 0.5
# start_step = 3750
# end_step = 4000

# vel = 1.2
# start_step = 6500
# end_step = 6800

# all
start_step = 500
end_step = 6000

# === Load & Filter ===
df = pd.read_csv(log_path)
df = df[(df["step"] >= start_step) & (df["step"] <= end_step)]

# === Plot ===
# plt.figure(figsize=(12, 5))
# contact_legs = ["LF", "RF", "LH", "RH"]
# offsets = [3, 2, 1, 0]

# for leg, offset in zip(contact_legs, offsets):
#     on_ground = df[df[leg] == 1]
#     plt.scatter(on_ground["step"], [offset]*len(on_ground), label=leg, s=8)

# # Overlay vx command
# plt.plot(df["step"], df["vx_cmd"], color='black', linewidth=1.5, label="vx_cmd")

# plt.yticks(offsets + [4], contact_legs + ["vx_cmd"])
# plt.xlabel("Timestep")
# plt.title("Foot Contact Pattern vs Command Velocity")
# plt.grid(True)
# plt.legend(loc="upper right")
# plt.tight_layout()
# plt.savefig(log_path.replace("_log.csv", f"_plot_{start_step}_{end_step}.png"))
# plt.show()


# === Plot Velocity Tracking ===
plt.figure(figsize=(12, 4))
plt.plot(df["step"], df["vx_cmd"], label="vx_cmd", color="black", linestyle="--")
plt.plot(df["step"], df["vx_actual"], label="vx_actual", color="blue")

# Optional: plot shaded error
error = df["vx_cmd"] - df["vx_actual"]
plt.fill_between(df["step"], df["vx_cmd"], df["vx_actual"], color='red', alpha=0.2, label="tracking error")

plt.xlabel("Timestep")
plt.ylabel("Velocity (m/s)")
plt.title("Forward Velocity Tracking")

# Mean error
mean_error = error.abs().mean()
plt.text(df["step"].iloc[0], max(df["vx_cmd"].max(), df["vx_actual"].max()) + 0.05,
         f"Mean abs error: {mean_error:.3f} m/s", fontsize=10)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(log_path.replace("_log.csv", f"_vxtrack_{start_step}_{end_step}.png"))
plt.show()



# === Plot Roll and pitch Stability ===
plt.figure(figsize=(12, 4))
plt.plot(df["step"], np.rad2deg(df["roll"]), label="roll (degree)", color="purple")
plt.plot(df["step"], np.rad2deg(df["pitch"]), label="pitch (degree)", color="green", linestyle="--")

mean_abs_roll = np.rad2deg(df["roll"].abs().mean())
mean_abs_pitch = np.rad2deg(df["pitch"].abs().mean())

plt.title(f"Torso Stability — Mean |roll|={mean_abs_roll:.3f} degree, |pitch|={mean_abs_pitch:.3f} degree")
plt.xlabel("Timestep")
plt.ylabel("Angle (degrees)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(log_path.replace("_log.csv", f"_orientation_{start_step}_{end_step}.png"))
plt.show()