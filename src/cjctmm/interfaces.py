"""Single interfaces: Snell's law, optical admittances and the Fresnel coefficients.

Everything here describes one flat interface between two homogeneous media.
The transfer-matrix solver in :mod:`cjctmm.tmm` is built from the same
functions (:func:`admittance`, :func:`snell_cos_theta` and the private
tangential-field helpers below), so the physics of a boundary lives in one
place only and a single-interface stack reproduces the Fresnel coefficients
exactly.

Conventions (see THEORY.md for the derivation):

- Time dependence ``exp(-i*omega*t)``; complex refractive index
  ``n = n' + i*kappa`` with ``kappa > 0`` for an absorbing medium.
- ``s`` (German *senkrecht*, perpendicular) = TE: the electric field is
  perpendicular to the plane of incidence. ``p`` (*parallel*) = TM: the
  electric field lies in the plane of incidence.
- ``r`` and ``t`` are complex amplitude ratios of the electric field;
  ``R`` and ``T`` are power fractions.
- p sign convention as in Hecht, Born & Wolf and Wikipedia: the magnetic
  fields of the incident, reflected and transmitted waves all point along
  the same direction (normal to the plane of incidence), and each electric
  field follows from ``E = Z * H x k_hat``. Then
  ``r_p = (n2 cos1 - n1 cos2) / (n2 cos1 + n1 cos2)``, which is ``+0.2`` for
  air to glass (n = 1.5) at normal incidence while ``r_s = -0.2``.

Angles are in radians.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt

from cjctmm._constants import VACUUM_IMPEDANCE

Polarization = Literal["s", "p"]

# Refractive indices, cosines and angles: complex scalars or arrays. numpy's ArrayLike
# already allows complex, but some type checkers (PyCharm) narrow it to float after a real
# argument and then reject a complex one; naming complex explicitly avoids that.
ComplexArrayLike = npt.ArrayLike | complex


def _check_polarization(polarization: str) -> None:
    """Raise a clear error unless ``polarization`` is ``"s"`` or ``"p"``."""
    if polarization not in ("s", "p"):
        raise ValueError(
            f"polarization must be 's' (senkrecht, TE) or 'p' (parallel, TM), got {polarization!r}"
        )


def admittance(
    n: ComplexArrayLike,
    cos_theta: ComplexArrayLike,
    polarization: Polarization,
) -> npt.NDArray[np.complex128]:
    """
    Tilted optical admittance of a medium for one polarization [S].

    Parameters
    ----------
    n:
        Complex refractive index ``n' + i*kappa``.
    cos_theta:
        Cosine of the propagation angle in the medium, from :func:`snell_cos_theta`.
    polarization:
        ``"s"`` (senkrecht = TE, E perpendicular to the plane of incidence) or
        ``"p"`` (parallel = TM, E in the plane of incidence).

    Returns
    -------
    numpy.ndarray of complex
        ``eta = n*cos(theta)/Z0`` for s, ``eta = n/(Z0*cos(theta))`` for p,
        broadcast over the inputs.

    Notes
    -----
    At an interface only the field components *parallel* to it are continuous,
    so the transfer-matrix method works with tangential fields. The admittance
    is the ratio ``H_tangential / E_tangential`` of a single plane wave. For a
    plane wave ``|H| = n|E|/Z0``. For s, E is already tangential while only
    ``H*cos(theta)`` is, giving ``n*cos(theta)/Z0``. For p, H is tangential
    while only ``E*cos(theta)`` is, giving ``n/(Z0*cos(theta))``. At normal
    incidence both reduce to ``n/Z0``.
    """

    # Validate first, so a typo like "P" or the old integer selector fails loudly.
    _check_polarization(polarization)

    # Complex arrays so lossy media and evanescent angles go through unchanged.
    n = np.asarray(n, dtype=np.complex128)
    cos_theta = np.asarray(cos_theta, dtype=np.complex128)

    # Tangential H over tangential E, see Notes.
    if polarization == "s":
        return n * cos_theta / VACUUM_IMPEDANCE
    return n / (VACUUM_IMPEDANCE * cos_theta)


def snell_cos_theta(
    n_inc: ComplexArrayLike,
    n: ComplexArrayLike,
    theta_inc: ComplexArrayLike,
) -> npt.NDArray[np.complex128]:
    """
    Cosine of the propagation angle in a medium, from Snell's law.

    Parameters
    ----------
    n_inc:
        Complex refractive index of the incident medium.
    n:
        Complex refractive index of the medium of interest.
    theta_inc:
        Angle of incidence in the incident medium [rad]. Real for a lossless
        incident medium; complex when the "incident medium" is an absorbing
        layer inside a stack (see :func:`snell_theta`).

    Returns
    -------
    numpy.ndarray of complex
        ``cos(theta)`` in the medium, on the branch with ``Im(cos theta) >= 0``
        (up to rounding, see Notes), broadcast over the inputs.

    Notes
    -----
    The component of the wavevector along the interface, ``(omega/c)*n*sin(theta)``,
    is the same in every layer of a planar stack (phase matching at every
    boundary). So ``sin(theta) = n_inc*sin(theta_inc)/n`` follows directly from
    the incident medium, without chaining layer by layer, which would add
    rounding error in a stack of many layers.

    The transfer-matrix method only ever needs ``cos(theta)``: it sets the
    wavevector component normal to the layers, ``k_z = (omega/c)*n*cos(theta)``,
    which gives the phase a wave picks up crossing a layer, and it tilts the
    admittance. The angle itself is never needed, and computing it first
    (via a complex arccos or logarithm) and then taking its cosine would only
    add rounding error and a chance to land on the wrong branch.

    ``cos(theta) = sqrt(1 - sin(theta)**2)`` has two roots. The one with
    ``Im(cos theta) >= 0`` makes ``exp(i*k_z*z)`` decay along +z: in an
    absorbing layer the forward wave is attenuated, and beyond the critical
    angle of total internal reflection ``cos(theta)`` is purely imaginary and
    the transmitted wave is evanescent. For passive media
    (``n' >= 0, kappa >= 0``) and a real Snell invariant this branch also
    has ``Im(n*cos theta) >= 0``, so both common branch rules agree.

    For such media the principal square root already lies on this branch
    (``Re >= 0`` and ``Im >= 0``). Rounding can still push it just across the
    real or imaginary axis, and the two cases need opposite treatment:

    - a propagating wave, ``cos(theta) ~ a - i*1e-17``: the tiny negative
      imaginary part is noise; flipping the sign would turn the wave around,
      so the root is kept;
    - an evanescent wave, ``sqrt(-x - 0j) = -i*sqrt(x)``: here the sign of
      the imaginary part is the physics, so the root is flipped to ``+i*sqrt(x)``.

    Hence a root is flipped only when it lies below the real axis *and* closer
    to the imaginary axis than to the real axis (``Im < -|Re|``). Gain media
    (``kappa < 0``) are outside the scope of this rule.
    """

    # Complex inputs so lossy media and evanescent waves are handled.
    n_inc = np.asarray(n_inc, dtype=np.complex128)
    n = np.asarray(n, dtype=np.complex128)
    theta_inc = np.asarray(theta_inc, dtype=np.complex128)

    # The transverse wavevector is conserved: sin(theta) from the incident medium.
    sin_theta = n_inc * np.sin(theta_inc) / n
    cos_theta = np.sqrt(1.0 - sin_theta * sin_theta)

    # Take the root whose wave decays into the medium, robust to rounding, see Notes.
    return np.where(cos_theta.imag < -np.abs(cos_theta.real), -cos_theta, cos_theta)


def snell_theta(
    n_inc: ComplexArrayLike,
    n: ComplexArrayLike,
    theta_inc: ComplexArrayLike,
) -> npt.NDArray[np.complex128]:
    """
    Complex propagation angle in a medium, from Snell's law.

    Not used by the solver, which needs only ``cos(theta)`` (see
    :func:`snell_cos_theta`); provided for plotting and checks.

    Parameters
    ----------
    n_inc:
        Complex refractive index of the incident medium.
    n:
        Complex refractive index of the medium of interest.
    theta_inc:
        Angle of incidence in the incident medium [rad].

    Returns
    -------
    numpy.ndarray of complex
        ``theta`` [rad]; real for a lossless medium below the critical angle.

    Notes
    -----
    In the simple case (real indices, below the critical angle) ``theta`` is
    real and ``arcsin`` or ``arccos`` would do. In general ``sin(theta)`` is
    complex, and ``arcsin`` alone cannot tell which of the two roots of
    ``cos(theta)`` belongs to the physical (decaying) wave. So both are fixed
    first: ``sin(theta) = n_inc*sin(theta_inc)/n`` from the Snell invariant,
    ``cos(theta)`` from :func:`snell_cos_theta` on the decaying branch, and
    then ``theta = -i*ln(cos(theta) + i*sin(theta))``, which follows from
    ``exp(i*theta) = cos(theta) + i*sin(theta)``. Beyond the critical angle
    this gives ``theta = pi/2 - i*arccosh(sin(theta))``.
    """

    # Both sine and the decaying-branch cosine fix the angle unambiguously.
    sin_theta = np.asarray(n_inc, dtype=np.complex128) * np.sin(theta_inc) / np.asarray(n, dtype=np.complex128)
    cos_theta = snell_cos_theta(n_inc, n, theta_inc)

    # Invert exp(i*theta) = cos(theta) + i*sin(theta).
    return -1j * np.log(cos_theta + 1j * sin_theta)


def critical_angle(n_1: npt.ArrayLike, n_2: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """
    Critical angle of total internal reflection from medium 1 into medium 2 [rad].

    Parameters
    ----------
    n_1, n_2:
        Real refractive indices of the incident medium and the medium behind the interface.

    Returns
    -------
    numpy.ndarray of float
        ``arcsin(n_2/n_1)``; NaN where ``n_2 >= n_1``, because then no angle
        gives total internal reflection.

    Notes
    -----
    Beyond this angle ``n_1*sin(theta_1) > n_2``, so ``sin(theta_2) > 1`` and
    ``cos(theta_2)`` becomes purely imaginary: the wave in medium 2 is
    evanescent, carries no power away, and ``R = 1``.
    """

    # Real ratio; arcsin is only defined for a ratio below 1.
    ratio = np.asarray(n_2, dtype=np.float64) / np.asarray(n_1, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.where(ratio < 1.0, np.arcsin(np.minimum(ratio, 1.0)), np.nan)


def brewster_angle(n_1: npt.ArrayLike, n_2: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """
    Brewster angle from medium 1 onto medium 2 [rad].

    Parameters
    ----------
    n_1, n_2:
        Real refractive indices of the incident medium and the medium behind the interface.

    Returns
    -------
    numpy.ndarray of float
        ``arctan(n_2/n_1)``, the angle at which ``r_p = 0``.

    Notes
    -----
    At this angle the reflected and transmitted rays are perpendicular, so the
    dipoles driven in medium 2 oscillate along the would-be reflected ray and
    cannot radiate into it: p-polarised light is not reflected at all, and
    ``r_p`` changes sign across this angle.
    """

    # Exact for real indices; for absorbing media only a reflectance minimum remains.
    return np.arctan(np.asarray(n_2, dtype=np.float64) / np.asarray(n_1, dtype=np.float64))


def _tangential_r(
    eta_1: npt.NDArray[np.complex128],
    eta_2: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Ratio of reflected to incident tangential E at an interface 1 -> 2, from admittances."""
    # Continuity of tangential E and H: E_i + E_r = E_t and eta_1*(E_i - E_r) = eta_2*E_t.
    return (eta_1 - eta_2) / (eta_1 + eta_2)


def _tangential_t(
    eta_1: npt.NDArray[np.complex128],
    eta_2: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Ratio of transmitted to incident tangential E at an interface 1 -> 2, from admittances."""
    # Same two boundary conditions as _tangential_r, solved for E_t.
    return 2.0 * eta_1 / (eta_1 + eta_2)


def _r_from_tangential(
    r_tangential: npt.NDArray[np.complex128],
    polarization: Polarization,
) -> npt.NDArray[np.complex128]:
    """Convert a tangential-E reflection ratio to the textbook r_s or r_p."""
    # For s the whole E field is tangential. For p, with H of both waves along
    # the same direction, the tangential parts of the incident and reflected E
    # point in opposite directions (E = Z*H x k_hat and k_z flips sign), so
    # r_p = -r_tangential.
    if polarization == "s":
        return r_tangential
    return -r_tangential


def _t_from_tangential(
    t_tangential: npt.NDArray[np.complex128],
    cos_theta_inc: npt.NDArray[np.complex128],
    cos_theta_exit: npt.NDArray[np.complex128],
    polarization: Polarization,
) -> npt.NDArray[np.complex128]:
    """Convert a tangential-E transmission ratio to the textbook t_s or t_p."""
    # For p the tangential part of E is E*cos(theta) in each medium, so the full
    # field ratio is t_tangential * cos(theta_inc) / cos(theta_exit).
    if polarization == "s":
        return t_tangential
    return t_tangential * cos_theta_inc / cos_theta_exit


def fresnel_r(
    n_1: ComplexArrayLike,
    n_2: ComplexArrayLike,
    theta_inc: ComplexArrayLike,
    polarization: Polarization = "s",
) -> npt.NDArray[np.complex128]:
    """
    Fresnel amplitude reflection coefficient of one interface.

    Parameters
    ----------
    n_1, n_2:
        Complex refractive indices of the incident medium and the medium behind the interface.
    theta_inc:
        Angle of incidence in medium 1 [rad]; complex for an interface behind
        an absorbing layer, where the angle is complex (see :func:`snell_theta`).
    polarization:
        ``"s"`` (senkrecht = TE) or ``"p"`` (parallel = TM).

    Returns
    -------
    numpy.ndarray of complex
        ``r_s = (n1 cos1 - n2 cos2)/(n1 cos1 + n2 cos2)`` or
        ``r_p = (n2 cos1 - n1 cos2)/(n2 cos1 + n1 cos2)`` (Hecht convention, see
        the module docstring).

    Notes
    -----
    Written in admittances, ``r = (eta_1 - eta_2)/(eta_1 + eta_2)`` for the
    tangential E field, which is exactly r_s. For p that tangential ratio is
    ``-r_p`` (see ``_r_from_tangential``). For the opposite p sign convention
    use ``-fresnel_r(..., polarization="p")``.
    """

    # Admittances on both sides, from Snell's law (the incident side reproduces cos(theta_inc)).
    eta_1 = admittance(n_1, snell_cos_theta(n_1, n_1, theta_inc), polarization)
    eta_2 = admittance(n_2, snell_cos_theta(n_1, n_2, theta_inc), polarization)

    # The tangential ratio from admittances, converted to the textbook coefficient.
    return _r_from_tangential(_tangential_r(eta_1, eta_2), polarization)


def fresnel_t(
    n_1: ComplexArrayLike,
    n_2: ComplexArrayLike,
    theta_inc: ComplexArrayLike,
    polarization: Polarization = "s",
) -> npt.NDArray[np.complex128]:
    """
    Fresnel amplitude transmission coefficient of one interface.

    Parameters
    ----------
    n_1, n_2:
        Complex refractive indices of the incident medium and the medium behind the interface.
    theta_inc:
        Angle of incidence in medium 1 [rad]; complex for an interface behind
        an absorbing layer, where the angle is complex (see :func:`snell_theta`).
    polarization:
        ``"s"`` (senkrecht = TE) or ``"p"`` (parallel = TM).

    Returns
    -------
    numpy.ndarray of complex
        ``t_s = 2 n1 cos1/(n1 cos1 + n2 cos2)`` or
        ``t_p = 2 n1 cos1/(n2 cos1 + n1 cos2)``, ratios of the full electric
        field amplitudes.

    Notes
    -----
    Written in admittances, ``t = 2*eta_1/(eta_1 + eta_2)`` for the tangential
    E field. For p the tangential part of E is ``E*cos(theta)`` on each side,
    so ``t_p`` carries the extra factor ``cos1/cos2``.
    """

    # Cosines and admittances on both sides, from Snell's law.
    cos_1 = snell_cos_theta(n_1, n_1, theta_inc)
    cos_2 = snell_cos_theta(n_1, n_2, theta_inc)
    eta_1 = admittance(n_1, cos_1, polarization)
    eta_2 = admittance(n_2, cos_2, polarization)

    # The tangential ratio from admittances, converted to the full-field coefficient.
    return _t_from_tangential(_tangential_t(eta_1, eta_2), cos_1, cos_2, polarization)


def fresnel_R(
    n_1: ComplexArrayLike,
    n_2: ComplexArrayLike,
    theta_inc: ComplexArrayLike,
    polarization: Polarization = "s",
) -> npt.NDArray[np.float64]:
    """
    Fresnel reflectance (reflected power fraction) of one interface.

    Parameters
    ----------
    n_1, n_2, theta_inc, polarization:
        As in :func:`fresnel_r`.

    Returns
    -------
    numpy.ndarray of float
        ``R = |r|^2``, the same for either p sign convention.
    """

    # Incident and reflected waves travel in the same medium, so R is just |r|^2.
    return np.abs(fresnel_r(n_1, n_2, theta_inc, polarization)) ** 2


def fresnel_T(
    n_1: ComplexArrayLike,
    n_2: ComplexArrayLike,
    theta_inc: ComplexArrayLike,
    polarization: Polarization = "s",
) -> npt.NDArray[np.float64]:
    """
    Fresnel transmittance (transmitted power fraction) of one interface.

    Parameters
    ----------
    n_1, n_2, theta_inc, polarization:
        As in :func:`fresnel_r`. Medium 1 should be lossless for ``R + T = 1``.

    Returns
    -------
    numpy.ndarray of float
        ``T = Re(eta_2)/Re(eta_1) * |t_tangential|^2``: the ratio of the power
        flows normal to the interface. Not simply ``|t|^2``, because the waves
        travel in different media and at different angles.

    Notes
    -----
    The time-averaged power flow normal to the interface of a single wave is
    ``Re(E_tangential * conj(H_tangential))/2 = Re(eta)*|E_tangential|^2/2``.
    Beyond the critical angle ``eta_2`` is purely imaginary and ``T = 0``.
    """

    # Admittances on both sides, from Snell's law.
    eta_1 = admittance(n_1, snell_cos_theta(n_1, n_1, theta_inc), polarization)
    eta_2 = admittance(n_2, snell_cos_theta(n_1, n_2, theta_inc), polarization)

    # Power flow normal to the interface on each side, from the tangential fields.
    return np.real(eta_2) / np.real(eta_1) * np.abs(_tangential_t(eta_1, eta_2)) ** 2
