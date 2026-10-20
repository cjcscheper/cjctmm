"""Fabry-Perot slab: transmission of a free-standing lossless slab, T = 1 where 2 n d = m * wavelength."""

from pathlib import Path

import numpy as np
from cjcplot import plot_1d

from cjctmm import SPEED_OF_LIGHT, solve_tmm

# A 500 um slab with n = 3.4 in vacuum: resonances every c/(2 n d) ~ 88 GHz.
n_slab, d = 3.4, 500e-6
frequency = np.linspace(0.05e12, 0.5e12, 2000)
omega = 2 * np.pi * frequency

# Fully coherent, and with only the direct pass (as a time window that cuts off all echoes).
coherent = solve_tmm(omega, [np.nan, d, np.nan], [1.0, n_slab, 1.0])
single_pass = solve_tmm(omega, [np.nan, d, np.nan], [1.0, n_slab, 1.0], num_substrate_passes=1)
resonances_thz = np.arange(1, 6) * SPEED_OF_LIGHT / (2 * n_slab * d) * 1e-12

frequency_thz = frequency * 1e-12
# The resonances as markers at T = 1, where the coherent curve must peak.
plot_1d(
    [frequency_thz, frequency_thz, resonances_thz],
    [coherent.T, single_pass.T, np.ones_like(resonances_thz)],
    line_styles=["-", "--"],
    line_show=[True, True, False],
    scatter_show=[False, False, True],
    legend_labels=["all echoes", "direct pass only", r"$2nd = m\lambda$"],
    xlim=(frequency_thz[0], frequency_thz[-1]),
    ylim=(0, 1.45),
    xlabel="Frequency [THz]",
    ylabel="Transmittance",
    legend_loc="upper center",
    legend_ncols=2,
    fig_aspect_ratio=(0.75, 0.5),
    save_path=str(Path(__file__).with_suffix(".png")),
    fig_show=False,
)
