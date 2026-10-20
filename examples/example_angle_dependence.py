"""R_s and R_p versus angle of incidence: Brewster angle, and total internal reflection beyond the critical angle.

Saves two figures: from air into glass, and from glass into air.
"""

from pathlib import Path

import numpy as np
from cjcplot import plot_1d

from cjctmm import brewster_angle, critical_angle, fresnel_R

# One interface between air and glass, seen from both sides.
n_glass = 1.5
theta = np.radians(np.linspace(0.0, 89.9, 900))
theta_deg = np.degrees(theta)

# The legend goes where each figure has empty space.
for n_1, n_2, name, legend_loc in [
    (1.0, n_glass, "air_to_glass", "upper left"),
    (n_glass, 1.0, "glass_to_air", "center right"),
]:
    # Brewster: R_p = 0. Critical angle: only when going to the optically thinner medium.
    lines_deg = [np.degrees(brewster_angle(n_1, n_2))]
    lines_labels = [f"Brewster {lines_deg[0]:.1f}°"]
    theta_c = critical_angle(n_1, n_2)
    if np.isfinite(theta_c):
        lines_deg.append(np.degrees(theta_c))
        lines_labels.append(f"critical {lines_deg[1]:.1f}°")

    plot_1d(
        [theta_deg, theta_deg],
        [fresnel_R(n_1, n_2, theta, "s"), fresnel_R(n_1, n_2, theta, "p")],
        legend_labels=[r"$R_s$", r"$R_p$"],
        vlines_pos=lines_deg,
        vlines_styles=[":", "--"],
        vlines_labels=lines_labels,
        xlim=(0, 90),
        ylim=(0, 1.02),
        xlabel="Angle of incidence [deg]",
        ylabel="Reflectance",
        legend_loc=legend_loc,
        fig_aspect_ratio=(0.75, 0.5),
        save_path=str(Path(__file__).with_name(f"{Path(__file__).stem}_{name}.png")),
        fig_show=False,
    )
