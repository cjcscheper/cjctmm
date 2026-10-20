"""Quarter-wave antireflection coating on glass, designed for 550 nm."""

from pathlib import Path

import numpy as np
from cjcplot import plot_1d

from cjctmm import SPEED_OF_LIGHT, solve_tmm

output_stem = Path(__file__).with_suffix("")

# Glass substrate and the ideal single-layer coating index sqrt(n_glass).
n_glass = 1.5
n_coating = np.sqrt(n_glass)
wavelength = np.linspace(400e-9, 800e-9, 400)
omega = 2 * np.pi * SPEED_OF_LIGHT / wavelength

# Bare glass, and glass with a quarter-wave layer at 550 nm: optical thickness n*d = 550 nm / 4.
bare = solve_tmm(omega, [np.nan, np.nan], [1.0, n_glass])
coated = solve_tmm(omega, [np.nan, 550e-9 / (4 * n_coating), np.nan], [1.0, n_coating, n_glass])
plot_1d(
    [wavelength * 1e9, wavelength * 1e9],
    [bare.R * 100, coated.R * 100],
    legend_labels=["bare", "AR-coated"],
    vlines_pos=[550.0], vlines_styles=[":"], vlines_labels=[r"$\lambda_0$"],
    xlabel="Wavelength [nm]", ylabel="Reflectance [%]",
    xlim=(400.0, 800.0), ylim=(0.0, 5.0),
    legend_loc="center right",
    fig_aspect_ratio=(0.5, 0.5),
    save_path=f"{output_stem}.png",
    fig_show=False,
)
