"""Fabry-Perot slab: a freestanding lossless slab transmits fully where 2 n d = m * wavelength."""

from pathlib import Path

import numpy as np
from cjcplot import plot_1d

from cjctmm import solve_tmm

output_stem = Path(__file__).with_suffix("")

frequency = np.linspace(1e9, 1e12, 2000)
omega = 2 * np.pi * frequency

# A 200 um slab with n = 3.4 in vacuum (resonances every c/(2 n d) ~ 0.22 THz): all echoes
# (coherent, infinitely many passes) against only the direct pass (a time window that cuts off every echo).
coherent = solve_tmm(omega, [np.nan, 200e-6, np.nan], [1.0, 3.4, 1.0])
single_pass = solve_tmm(omega, [np.nan, 200e-6, np.nan], [1.0, 3.4, 1.0], num_substrate_passes=1)
plot_1d(
    [frequency * 1e-12, frequency * 1e-12],
    [coherent.T, single_pass.T],
    line_styles=["-", "--"],
    legend_labels=[r"$N_\mathrm{passes} = \infty$", r"$N_\mathrm{passes} = 1$"],
    xlabel="Frequency [THz]", ylabel="Transmittance",
    xlim=(0.0, 1.0), ylim=(0.0, 1.0),
    legend_loc="lower center",
    fig_aspect_ratio=(0.5, 0.5),
    save_path=f"{output_stem}.png",
    fig_show=False,
)
