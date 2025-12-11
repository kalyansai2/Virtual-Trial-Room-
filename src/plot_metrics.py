import numpy as np
import matplotlib.pyplot as plt

# Your metrics
metrics = {
    "FID": 12.91,
    "KID": 0.810,
    "SSIM": 0.832,
    "LPIPS": 0.068
}

# ------- BAR CHART -------
names = list(metrics.keys())
values = list(metrics.values())

plt.figure(figsize=(8, 5))
bars = plt.bar(names, values)

# Color bars based on metric type
bars[0].set_color("steelblue")   # FID
bars[1].set_color("mediumpurple") # KID
bars[2].set_color("seagreen")     # SSIM
bars[3].set_color("tomato")       # LPIPS

plt.title("CatVTON Evaluation Metrics", fontsize=16)
plt.ylabel("Value", fontsize=13)

for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=12)

plt.tight_layout()
plt.savefig("metrics_bar_chart.png", dpi=300)
print("Saved metrics_bar_chart.png")


# ------- RADAR CHART -------
# Normalize for visualization
max_vals = {
    "FID": 50,    # typical range 0–50
    "KID": 1.0,   # 0–1
    "SSIM": 1.0,  # 0–1
    "LPIPS": 0.3  # 0–0.3
}

normalized = [metrics[k]/max_vals[k] for k in names]

angles = np.linspace(0, 2*np.pi, len(names), endpoint=False)
normalized += normalized[:1]
angles = np.concatenate((angles, [angles[0]]))

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.plot(angles, normalized, "o-", linewidth=2)
ax.fill(angles, normalized, alpha=0.3)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(names)

plt.title("CatVTON Metrics (Normalized Radar Plot)", fontsize=16)
plt.tight_layout()
plt.savefig("metrics_radar_chart.png", dpi=300)
print("Saved metrics_radar_chart.png")
