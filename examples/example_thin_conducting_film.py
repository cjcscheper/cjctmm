"""Thin conducting film on a substrate, against the thin-film (Tinkham) formula."""

from pathlib import Path

import numpy as np
from cjcplot import plot_1d, style_colors

from cjctmm import VACUUM_IMPEDANCE, VACUUM_PERMITTIVITY, solve_tmm

# A 10 nm Drude metal film (scattering time 100 fs) on a semi-infinite substrate.
n_substrate, d, tau = 3.4, 10e-9, 100e-15
frequency = np.linspace(0.1e12, 5e12, 300)
omega = 2 * np.pi * frequency
sigma_dc_values = [1e4, 1e5, 1e6]

# Transmission relative to the bare substrate, as measured with a substrate reference.
bare = solve_tmm(omega, [np.nan, np.nan], [1.0, n_substrate])
tmm_ratios, tinkham_ratios = [], []
for sigma_dc in sigma_dc_values:
    # Drude conductivity for time dependence exp(-i omega t), and the film index from its permittivity.
    sigma = sigma_dc / (1 - 1j * omega * tau)
    n_film = np.sqrt(1.0 + 1j * sigma / (VACUUM_PERMITTIVITY * omega))
    layer_n = np.column_stack([np.ones_like(omega), n_film, np.full_like(omega, n_substrate)])
    film = solve_tmm(omega, [np.nan, d, np.nan], layer_n)
    tmm_ratios.append(np.abs(film.t / bare.t))
    tinkham_ratios.append(np.abs((1 + n_substrate) / (1 + n_substrate + VACUUM_IMPEDANCE * sigma * d)))

# TMM as lines, the Tinkham formula as markers on every 15th point, both in the same colour per film.
# The last trace is an invisible (NaN) grey marker that gives the formula one neutral legend entry.
frequency_thz = frequency * 1e-12
count = len(sigma_dc_values)
colors = style_colors()[:count]
plot_1d(
    [frequency_thz] * count + [frequency_thz[::15]] * count + [[np.nan]],
    tmm_ratios + [ratio[::15] for ratio in tinkham_ratios] + [[np.nan]],
    colors=colors + colors + ["gray"],
    line_show=[True] * count + [False] * (count + 1),
    scatter_show=[False] * count + [True] * (count + 1),
    legend_labels=[rf"$\sigma_\mathrm{{DC}}$ = $10^{{{np.log10(s):.0f}}}$ S/m" for s in sigma_dc_values]
    + [""] * count
    + ["Tinkham formula"],
    xlim=(0, 5),
    ylim=(0, 1.05),
    xlabel="Frequency [THz]",
    ylabel=r"$|t_\mathrm{film}/t_\mathrm{substrate}|$",
    legend_loc="lower right",
    fig_aspect_ratio=(0.75, 0.5),
    save_path=str(Path(__file__).with_suffix(".png")),
    fig_show=False,
)
