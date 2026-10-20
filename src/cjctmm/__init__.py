"""cjctmm: transfer-matrix method for multilayer optics.

Arrays in, a frozen :class:`Result` out: give the angular frequencies, the
layer thicknesses and the complex refractive indices of a planar stack, and
:func:`solve_tmm` returns its complex reflection and transmission
coefficients and the reflected, transmitted and absorbed power fractions.
The single-interface physics (Snell's law, admittances, Fresnel
coefficients) is available on its own as well. numpy is the only dependency.
"""

from cjctmm._constants import (
    SPEED_OF_LIGHT,
    VACUUM_IMPEDANCE,
    VACUUM_PERMEABILITY,
    VACUUM_PERMITTIVITY,
)
from cjctmm.interfaces import (
    Polarization,
    admittance,
    brewster_angle,
    critical_angle,
    fresnel_r,
    fresnel_R,
    fresnel_t,
    fresnel_T,
    snell_cos_theta,
    snell_theta,
)
from cjctmm.tmm import MAX_IMAG_PHASE, Result, propagation_phase, solve_tmm

__version__ = "0.1.0"

__all__ = [
    "MAX_IMAG_PHASE",
    "Polarization",
    "Result",
    "SPEED_OF_LIGHT",
    "VACUUM_IMPEDANCE",
    "VACUUM_PERMEABILITY",
    "VACUUM_PERMITTIVITY",
    "admittance",
    "brewster_angle",
    "critical_angle",
    "fresnel_R",
    "fresnel_T",
    "fresnel_r",
    "fresnel_t",
    "propagation_phase",
    "snell_cos_theta",
    "snell_theta",
    "solve_tmm",
]
