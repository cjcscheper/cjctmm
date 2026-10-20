"""R_s and R_p versus angle of incidence: from air into glass (Brewster angle) and from glass into air (critical angle)."""

from pathlib import Path

import numpy as np
from cjcplot import plot_1d

from cjctmm import brewster_angle, critical_angle, fresnel_R

output_stem = Path(__file__).with_suffix("")

n_air = 1.0
n_glass = 1.5
theta = np.radians(np.linspace(0.0, 89.9, 900))
theta_deg = np.degrees(theta)

# Air into glass: R_p drops to zero at the Brewster angle.
brewster_deg = float(np.degrees(brewster_angle(n_air, n_glass)))
plot_1d(
    [theta_deg, theta_deg],
    [fresnel_R(n_air, n_glass, theta, "s"), fresnel_R(n_air, n_glass, theta, "p")],
    legend_labels=[r"$R_s$", r"$R_p$"],
    vlines_pos=[brewster_deg], vlines_styles=[":"], vlines_labels=[r"$\theta_B$"],
    xlabel="Angle of incidence [deg]", ylabel="Reflectance",
    xlim=(0.0, 90.0), ylim=(0.0, 1.0),
    legend_loc="upper left",
    fig_aspect_ratio=(0.5, 0.5),
    save_path=f"{output_stem}_air_to_glass.png",
    fig_show=False,
)

# Glass into air: Brewster angle, then total internal reflection beyond the critical angle.
brewster_deg = float(np.degrees(brewster_angle(n_glass, n_air)))
critical_deg = float(np.degrees(critical_angle(n_glass, n_air)))
plot_1d(
    [theta_deg, theta_deg],
    [fresnel_R(n_glass, n_air, theta, "s"), fresnel_R(n_glass, n_air, theta, "p")],
    legend_labels=[r"$R_s$", r"$R_p$"],
    vlines_pos=[brewster_deg, critical_deg], vlines_styles=[":", "--"],
    vlines_labels=[r"$\theta_B$", r"$\theta_c$"],
    xlabel="Angle of incidence [deg]", ylabel="Reflectance",
    xlim=(0.0, 90.0), ylim=(0.0, 1.0),
    legend_loc="center right",
    fig_aspect_ratio=(0.5, 0.5),
    save_path=f"{output_stem}_glass_to_air.png",
    fig_show=False,
)
