"""Quarter-wave antireflection coating on glass, designed for 550 nm."""

from pathlib import Path

import numpy as np
from cjcplot import plot_1d

from cjctmm import SPEED_OF_LIGHT, solve_tmm

# Glass substrate and the ideal single-layer coating index sqrt(n_glass).
n_glass = 1.5
n_coating = np.sqrt(n_glass)
design_wavelength = 550e-9

# Visible range; the solver works with angular frequency.
wavelength = np.linspace(400e-9, 800e-9, 400)
omega = 2 * np.pi * SPEED_OF_LIGHT / wavelength

# Bare glass, and glass with a quarter-wave layer: optical thickness n*d = wavelength/4.
bare = solve_tmm(omega, [np.nan, np.nan], [1.0, n_glass])
coated = solve_tmm(omega, [np.nan, design_wavelength / (4 * n_coating), np.nan], [1.0, n_coating, n_glass])

wavelength_nm = wavelength * 1e9
plot_1d(
    [wavelength_nm, wavelength_nm],
    [bare.R * 100, coated.R * 100],
    legend_labels=["bare glass", "quarter-wave coating"],
    vlines_pos=[design_wavelength * 1e9],
    vlines_styles=[":"],
    vlines_labels=["design wavelength"],
    xlim=(400, 800),
    ylim=(0, 5),
    xlabel="Wavelength [nm]",
    ylabel="Reflectance [%]",
    legend_loc="center right",
    fig_aspect_ratio=(0.75, 0.5),
    save_path=str(Path(__file__).with_suffix(".png")),
    fig_show=False,
)
