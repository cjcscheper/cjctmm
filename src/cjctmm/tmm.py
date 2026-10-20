"""Transfer-matrix method (TMM) for a planar multilayer stack.

The stack is a sequence of homogeneous, isotropic layers between two
semi-infinite media: light comes in from the first medium (index 0) and
leaves into the last (index N-1). Layer ``j`` has thickness ``d_j`` and
complex refractive index ``n_j(omega)``. The method follows Born & Wolf
(*Principles of Optics*, ch. 1.6) and Macleod (*Thin-Film Optical Filters*,
ch. 2); Hecht (*Optics*, ch. 4 and 9) covers the Fresnel and thin-film basics.

1. **Boundary conditions.** Without surface currents, the field components
   parallel to an interface, E_tangential and H_tangential, are continuous
   across it. Inside one layer the field is the sum of a forward and a
   backward plane wave. For a single plane wave ``H_tan = eta * E_tan``, with
   the tilted admittance ``eta = n cos(theta)/Z0`` (s) or ``n/(Z0 cos(theta))``
   (p), see :func:`cjctmm.admittance`. The angle follows from Snell's law,
   ``n sin(theta)`` being the same in every layer
   (:func:`cjctmm.snell_cos_theta`).

2. **Characteristic matrix per layer.** Writing the two waves in a layer in
   terms of the total tangential fields at its front (a) and back (b) face
   gives::

       [E_a]   [ cos(delta)          -i sin(delta)/eta ] [E_b]
       [H_a] = [ -i eta sin(delta)    cos(delta)       ] [H_b]

   with the phase thickness ``delta = (omega/c) n d cos(theta)``
   (:func:`propagation_phase`). A purely forward wave
   (``H_b = eta E_b``) indeed gives ``E_a = exp(-i delta) E_b``: it gained
   ``exp(+i delta)`` crossing the layer, as it should with time dependence
   ``exp(-i omega t)``.

3. **Product.** Because E_tan and H_tan are continuous at every interface,
   the matrices of all interior layers simply multiply in stack order:
   ``M = M_1 M_2 ... M_(N-2)``. Here the product is computed for all
   frequencies at once, pairwise (neighbours first), so a stack of many thin
   sublayers costs a handful of array operations.

4. **r and t.** In the exit medium there is only a transmitted wave, so
   ``[E; H]`` there is ``[1; eta_exit]`` times its tangential amplitude.
   With ``[B; C] = M [1; eta_exit]``, matching to the incident and reflected
   waves in the incident medium gives the tangential-field ratios::

       r_tan = (eta_inc B - C) / (eta_inc B + C)
       t_tan = 2 eta_inc / (eta_inc B + C)

   For s these are the textbook ``r_s`` and ``t_s``. For p they are converted
   to the textbook (Hecht / Born & Wolf) convention: ``r_p = -r_tan`` and
   ``t_p = t_tan cos(theta_inc)/cos(theta_exit)``, see THEORY.md.

5. **Power.** The time-averaged power flow normal to the layers of one wave
   is ``Re(eta) |E_tan|^2 / 2``. Hence ``R = |r|^2``,
   ``T = Re(eta_exit)/Re(eta_inc) |t_tan|^2`` and the absorbed fraction is
   ``A = 1 - R - T``. The incident medium should be lossless, otherwise the
   incident and reflected waves interfere in the power flow and R, T lose
   their usual meaning.

Conventions: time dependence ``exp(-i omega t)`` with ``n = n' + i kappa``,
``kappa > 0`` absorbing. ``conjugate_model=True`` returns ``r`` and ``t`` in
the opposite convention ``exp(+i omega t)``.
"""

from dataclasses import dataclass
import warnings

import numpy as np
import numpy.typing as npt

from cjctmm._constants import SPEED_OF_LIGHT
from cjctmm.interfaces import (
    ComplexArrayLike,
    Polarization,
    _check_polarization,
    _r_from_tangential,
    _t_from_tangential,
    _tangential_r,
    _tangential_t,
    admittance,
    snell_cos_theta,
)

# Largest |Im(delta)| used in cos/sin of a layer. A layer with Im(delta) = 40
# attenuates a single pass by exp(-40) ~ 4e-18 in amplitude, so it is opaque
# and clamping changes nothing measurable; without the clamp, cosh/sinh of a
# thick metal layer would overflow float64 (near exp(709)), also after the
# matrix products.
MAX_IMAG_PHASE = 40.0


@dataclass(frozen=True)
class Result:
    """
    Response of a multilayer stack, one value per frequency.

    Attributes
    ----------
    omega:
        Angular frequency [rad/s], shape ``(M,)``, as passed to :func:`solve_tmm`.
    r:
        Complex amplitude reflection coefficient, reflected over incident electric
        field in the incident medium, shape ``(M,)``. Textbook (Hecht) convention for p.
    t:
        Complex amplitude transmission coefficient, transmitted electric field in
        the exit medium over incident electric field, shape ``(M,)``.
    R:
        Reflectance, reflected power fraction ``|r|^2``, shape ``(M,)``.
    T:
        Transmittance, transmitted power fraction (power flow normal to the
        layers, not ``|t|^2``), shape ``(M,)``.
    A:
        Absorptance ``1 - R - T``, the power fraction absorbed in the stack, shape ``(M,)``.
    """

    omega: npt.NDArray[np.float64]
    r: npt.NDArray[np.complex128]
    t: npt.NDArray[np.complex128]
    R: npt.NDArray[np.float64]
    T: npt.NDArray[np.float64]
    A: npt.NDArray[np.float64]


def propagation_phase(
    omega: npt.ArrayLike,
    n: ComplexArrayLike,
    d: npt.ArrayLike,
    cos_theta: ComplexArrayLike,
) -> npt.NDArray[np.complex128]:
    """
    Phase thickness of a layer, ``delta = (omega/c) n d cos(theta)``.

    Parameters
    ----------
    omega:
        Angular frequency [rad/s].
    n:
        Complex refractive index of the layer.
    d:
        Layer thickness [m].
    cos_theta:
        Cosine of the propagation angle in the layer, from :func:`cjctmm.snell_cos_theta`.

    Returns
    -------
    numpy.ndarray of complex
        ``delta``, broadcast over the inputs (element-wise numpy broadcasting).

    Notes
    -----
    ``k0 = omega/c`` is the vacuum wavenumber and ``k0 n cos(theta)`` the
    wavevector component normal to the layer, so ``delta`` is the phase a
    plane wave gains in one crossing. ``Re(delta)`` sets interference
    (``Re(delta) = pi/2`` is a quarter-wave layer); ``Im(delta)`` is the
    amplitude attenuation of one crossing, ``|exp(i delta)| = exp(-Im delta)``.
    ``2 Re(delta)/omega`` is the round-trip time of a pulse through the layer
    (with the phase index; dispersion makes the group delay differ).
    """

    # Normal component of n*k_hat times the vacuum wavenumber, times thickness. The result is
    # already complex; astype (no copy) only states that for type checkers, which infer the
    # type of a numpy product from its float factors.
    k0 = np.asarray(omega, dtype=np.float64) / SPEED_OF_LIGHT
    delta = (
        np.asarray(n, dtype=np.complex128)
        * np.asarray(cos_theta, dtype=np.complex128)
        * k0
        * np.asarray(d, dtype=np.float64)
    )
    return delta.astype(np.complex128, copy=False)


def _clamp_imag_phase(delta: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Limit |Im(delta)| to MAX_IMAG_PHASE so opaque layers do not overflow."""
    # Only the imaginary part is clamped: the layer stays opaque, its phase is kept.
    return delta.real + 1j * np.clip(delta.imag, -MAX_IMAG_PHASE, MAX_IMAG_PHASE)


def _characteristic_matrix_product(
    delta: npt.NDArray[np.complex128],
    eta: npt.NDArray[np.complex128],
) -> tuple[npt.NDArray[np.complex128], ...]:
    """
    Product of the characteristic matrices of consecutive layers.

    Parameters
    ----------
    delta:
        Phase thickness per frequency and layer, shape ``(M, K)``, already clamped.
    eta:
        Admittance per frequency and layer, shape ``(M, K)``.

    Returns
    -------
    tuple of four ``(M,)`` complex arrays
        Elements ``(m11, m12, m21, m22)`` of ``M_1 M_2 ... M_K``; the identity
        for ``K = 0``.

    Notes
    -----
    All layer matrices are built at once and multiplied pairwise, neighbours
    first and in order, so ``K`` layers take ``log2(K)`` rounds of array
    operations instead of ``K`` Python loop iterations.
    """

    # Layer-first layout (K, M), so each round slices whole rows.
    cos_delta = np.cos(delta.T)
    sin_delta = np.sin(delta.T)
    eta = eta.T
    m11 = cos_delta
    m12 = -1j * sin_delta / eta
    m21 = -1j * eta * sin_delta
    m22 = cos_delta

    # No interior layers: the fields pass the (single) interface unchanged.
    num_freq = delta.shape[0]
    if m11.shape[0] == 0:
        ones = np.ones(num_freq, dtype=np.complex128)
        zeros = np.zeros(num_freq, dtype=np.complex128)
        return ones, zeros, zeros.copy(), ones.copy()

    # Multiply neighbouring matrices pairwise until one product remains.
    while m11.shape[0] > 1:
        # An odd count carries its last matrix over to the next round unchanged.
        num_pairs = m11.shape[0] // 2
        a = slice(0, 2 * num_pairs, 2)
        b = slice(1, 2 * num_pairs, 2)
        p11 = m11[a] * m11[b] + m12[a] * m21[b]
        p12 = m11[a] * m12[b] + m12[a] * m22[b]
        p21 = m21[a] * m11[b] + m22[a] * m21[b]
        p22 = m21[a] * m12[b] + m22[a] * m22[b]
        if m11.shape[0] % 2:
            p11 = np.concatenate([p11, m11[-1:]])
            p12 = np.concatenate([p12, m12[-1:]])
            p21 = np.concatenate([p21, m21[-1:]])
            p22 = np.concatenate([p22, m22[-1:]])
        m11, m12, m21, m22 = p11, p12, p21, p22

    return m11[0], m12[0], m21[0], m22[0]


def _tangential_r_t_from_matrix(
    eta_inc: npt.NDArray[np.complex128],
    m11: npt.NDArray[np.complex128],
    m12: npt.NDArray[np.complex128],
    m21: npt.NDArray[np.complex128],
    m22: npt.NDArray[np.complex128],
    eta_exit: npt.NDArray[np.complex128],
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Tangential-field r and t of a stack from its characteristic matrix (step 4 of the module docstring)."""
    # Fields at the front face for unit tangential E transmitted into the exit medium.
    b = m11 + m12 * eta_exit
    c = m21 + m22 * eta_exit

    # Split the front-face fields into incident and reflected waves.
    denominator = eta_inc * b + c
    return (eta_inc * b - c) / denominator, 2.0 * eta_inc / denominator


def _validate_inputs(
    omega: npt.ArrayLike,
    layer_thickness: npt.ArrayLike,
    layer_n: ComplexArrayLike,
    polarization: str,
    theta_inc: float,
    num_substrate_passes: int | None,
    substrate_pass_tolerance: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    """Check shapes and values of the solve_tmm inputs; return them as arrays."""
    # Polarization first: it is the most common typo and cheap to check.
    _check_polarization(polarization)

    # Frequencies: a 1D array of positive values.
    omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    if omega.ndim != 1:
        raise ValueError(f"omega must be a 1D array of angular frequencies [rad/s], got shape {omega.shape}")
    if not np.all(omega > 0.0):
        raise ValueError("omega must be positive (angular frequency in rad/s)")

    # Thicknesses: 1D, at least the two semi-infinite media.
    layer_thickness = np.asarray(layer_thickness, dtype=np.float64)
    if layer_thickness.ndim != 1 or layer_thickness.size < 2:
        raise ValueError(
            "layer_thickness must be a 1D array with at least the incident and exit media, "
            f"got shape {layer_thickness.shape}"
        )
    num_layers = layer_thickness.size

    # Refractive indices: (M, N), or (N,) for frequency-independent indices.
    layer_n = np.asarray(layer_n, dtype=np.complex128)
    if layer_n.shape == (num_layers,):
        layer_n = np.broadcast_to(layer_n, (omega.size, num_layers))
    if layer_n.shape != (omega.size, num_layers):
        raise ValueError(
            f"layer_n must have shape (len(omega), len(layer_thickness)) = ({omega.size}, {num_layers}) "
            f"or ({num_layers},) for frequency-independent indices, got {layer_n.shape}"
        )

    # Angle of incidence: real, from normal up to (not including) grazing.
    if not 0.0 <= theta_inc < np.pi / 2:
        raise ValueError(f"theta_inc must be in [0, pi/2) rad, got {theta_inc}")

    # Substrate passes: the substrate is the second-to-last layer and must be finite.
    if num_substrate_passes is not None:
        if num_layers < 3:
            raise ValueError(
                "num_substrate_passes needs at least 3 layers (incident medium, substrate, exit medium), "
                f"got {num_layers}"
            )
        if np.isnan(layer_thickness[-2]):
            raise ValueError(
                "num_substrate_passes needs a finite substrate thickness (layer_thickness[-2]), got nan"
            )
        if num_substrate_passes < 1:
            raise ValueError(f"num_substrate_passes must be >= 1, got {num_substrate_passes}")
        if not substrate_pass_tolerance > 0.0:
            raise ValueError(f"substrate_pass_tolerance must be positive, got {substrate_pass_tolerance}")

    # The two boundary media are semi-infinite: NaN marks that explicitly.
    if not (np.isnan(layer_thickness[0]) and np.isnan(layer_thickness[-1])):
        raise ValueError(
            "layer_thickness[0] and layer_thickness[-1] must be NaN: the incident and exit media are semi-infinite"
        )

    # Interior layers: finite, non-negative thicknesses.
    interior = layer_thickness[1:-1]
    bad = ~(np.isfinite(interior) & (interior >= 0.0))
    if np.any(bad):
        index = int(np.flatnonzero(bad)[0]) + 1
        raise ValueError(
            f"interior layer thicknesses must be finite and >= 0, got {layer_thickness[index]} at index {index}"
        )

    return omega, layer_thickness, layer_n


def _substrate_pass_error(
    omega: npt.NDArray[np.float64],
    layer_thickness: npt.NDArray[np.float64],
    layer_n: npt.NDArray[np.complex128],
    cos_theta: npt.NDArray[np.complex128],
    eta: npt.NDArray[np.complex128],
    num_substrate_passes: int,
) -> tuple[float, int]:
    """
    Estimate the error of truncating only the substrate echoes.

    Returns
    -------
    (float, int)
        The worst estimated relative amplitude error over frequencies and
        front layers, and the index of that layer (-1 if there are no front layers).

    Notes
    -----
    With ``num_substrate_passes = N`` the model keeps the substrate echoes up
    to ``N - 1`` round trips, as a measurement that is time-gated at
    ``t_gate = N * dt_sub`` would, with ``dt = 2 Re(delta)/omega`` the
    round-trip time of a layer. The front layers are still treated fully
    coherently, i.e. all their echoes are kept. A front layer ``j`` whose
    echoes arrive within the gate is fine; echoes beyond it, from round trip
    ``k_min = ceil(t_gate/dt_j)`` on, are kept by the model but would be gated
    away in the measurement. Each round trip multiplies an echo by
    ``rho = r_front r_back exp(2 i delta_j)`` (bare-interface Fresnel
    coefficients inside the layer), so their summed relative amplitude is
    about ``|rho|^k_min / (1 - |rho|)``.
    """

    # Gate time implied by the number of substrate passes, per frequency.
    substrate = layer_thickness.size - 2
    delta_substrate = propagation_phase(
        omega, layer_n[:, substrate], layer_thickness[substrate], cos_theta[:, substrate]
    )
    gate_time = num_substrate_passes * 2.0 * delta_substrate.real / omega

    # Check every front layer between the incident medium and the substrate.
    worst_error, worst_layer = 0.0, -1
    for j in range(1, substrate):
        # Round-trip time of the layer; an empty layer has no separate echoes.
        delta_j = propagation_phase(omega, layer_n[:, j], layer_thickness[j], cos_theta[:, j])
        round_trip_time = 2.0 * delta_j.real / omega
        has_echoes = round_trip_time > 0.0
        if not np.any(has_echoes):
            continue

        # Round-trip echo amplitude: reflections at both faces and the attenuation in between.
        r_front = _tangential_r(eta[:, j], eta[:, j - 1])
        r_back = _tangential_r(eta[:, j], eta[:, j + 1])
        rho = np.abs(r_front * r_back) * np.exp(-2.0 * _clamp_imag_phase(delta_j).imag)

        # Summed amplitude of the echoes beyond the gate, relative to the main pulse.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            k_min = np.ceil(gate_time[has_echoes] / round_trip_time[has_echoes])
            rho = rho[has_echoes]
            error = np.where(rho < 1.0, rho**k_min / (1.0 - rho), np.inf)

        # Keep the worst layer, for the warning message.
        layer_error = float(np.max(error))
        if layer_error > worst_error:
            worst_error, worst_layer = layer_error, j

    return worst_error, worst_layer


def solve_tmm(
    omega: npt.ArrayLike,
    layer_thickness: npt.ArrayLike,
    layer_n: ComplexArrayLike,
    *,
    polarization: Polarization = "s",
    theta_inc: float = 0.0,
    num_substrate_passes: int | None = None,
    substrate_pass_tolerance: float = 1e-3,
    conjugate_model: bool = False,
) -> Result:
    """
    Reflection and transmission of a multilayer stack by the transfer-matrix method.

    Parameters
    ----------
    omega:
        Angular frequency [rad/s], shape ``(M,)``. Any frequency range works:
        the method has no built-in scale.
    layer_thickness:
        Thickness of every layer [m], shape ``(N,)``, ordered from the incident
        medium to the exit medium. The first and last entries are the
        semi-infinite incident and exit media and must be NaN; interior layers
        are finite and ``>= 0``.
    layer_n:
        Complex refractive index ``n' + i kappa`` (``kappa > 0`` absorbing)
        per frequency and layer, shape ``(M, N)``; shape ``(N,)`` is accepted
        for frequency-independent indices. The incident medium should be lossless.
    polarization:
        ``"s"`` (German *senkrecht*, perpendicular) = TE: E perpendicular to
        the plane of incidence. ``"p"`` (*parallel*) = TM: E in the plane of
        incidence. Irrelevant at normal incidence.
    theta_inc:
        Angle of incidence in the incident medium [rad], in ``[0, pi/2)``.
    num_substrate_passes:
        ``None`` (default): fully coherent, all multiple reflections in every
        layer. An integer ``N >= 1``: keep only the first ``N`` passes through
        the substrate (the second-to-last layer) in transmission, and the
        matching echoes in reflection, as in a time-gated measurement whose
        window cuts off the later substrate echoes. ``N = 1`` is the direct
        pulse only. The front layers stay fully coherent.
    substrate_pass_tolerance:
        With ``num_substrate_passes``: warn (``UserWarning``) when the estimated
        relative amplitude error of keeping all front-layer echoes, including
        those that arrive after the gate, exceeds this value (see THEORY.md).
    conjugate_model:
        If True, return ``conj(r)`` and ``conj(t)``: the same response in the
        opposite time convention ``exp(+i omega t)`` (with ``n = n' - i kappa``).
        Inputs always use ``n' + i kappa``.

    Returns
    -------
    Result
        Frozen dataclass with ``omega, r, t, R, T, A``, each of shape ``(M,)``.
        For p, ``r`` and ``t`` follow the textbook convention
        ``r_p = (n2 cos1 - n1 cos2)/(n2 cos1 + n1 cos2)`` for a single
        interface; use ``-result.r`` for the opposite sign convention.

    Raises
    ------
    ValueError
        Wrong shapes, a polarization other than ``"s"``/``"p"``, an angle
        outside ``[0, pi/2)``, non-NaN boundary thicknesses, negative or NaN
        interior thicknesses; with ``num_substrate_passes``: fewer than 3
        layers, a NaN substrate thickness or fewer than 1 pass.

    Notes
    -----
    See the module docstring for the method and THEORY.md for the derivation.
    """

    # Check everything up front, so errors name the argument at fault.
    omega, layer_thickness, layer_n = _validate_inputs(
        omega,
        layer_thickness,
        layer_n,
        polarization,
        theta_inc,
        num_substrate_passes,
        substrate_pass_tolerance,
    )

    # Snell's law and admittances in every layer, from the incident medium's invariant.
    cos_theta = snell_cos_theta(layer_n[:, :1], layer_n, theta_inc)
    eta = admittance(layer_n, cos_theta, polarization)
    eta_inc, eta_exit = eta[:, 0], eta[:, -1]

    if num_substrate_passes is None:
        # Fully coherent: one characteristic matrix for all interior layers.
        delta = _clamp_imag_phase(
            propagation_phase(omega[:, None], layer_n[:, 1:-1], layer_thickness[1:-1], cos_theta[:, 1:-1])
        )
        m11, m12, m21, m22 = _characteristic_matrix_product(delta, eta[:, 1:-1])
        r_tan, t_tan = _tangential_r_t_from_matrix(eta_inc, m11, m12, m21, m22, eta_exit)
    else:
        # Warn when front-layer echoes outside the implied gate are not negligible.
        error, layer = _substrate_pass_error(
            omega, layer_thickness, layer_n, cos_theta, eta, num_substrate_passes
        )
        if error > substrate_pass_tolerance:
            warnings.warn(
                f"num_substrate_passes={num_substrate_passes}: front layer {layer} has echoes later than the "
                f"substrate gate; estimated relative amplitude error {error:.2e} exceeds "
                f"substrate_pass_tolerance={substrate_pass_tolerance:.1e}. The model truncates only the "
                "substrate echoes and keeps all echoes of the front layers.",
                UserWarning,
                stacklevel=2,
            )

        # Front stack: incident medium and front layers, with the substrate as its exit medium.
        substrate = layer_thickness.size - 2
        eta_substrate = eta[:, substrate]
        delta = _clamp_imag_phase(
            propagation_phase(
                omega[:, None], layer_n[:, 1:substrate], layer_thickness[1:substrate], cos_theta[:, 1:substrate]
            )
        )
        m11, m12, m21, m22 = _characteristic_matrix_product(delta, eta[:, 1:substrate])
        r_12, t_12 = _tangential_r_t_from_matrix(eta_inc, m11, m12, m21, m22, eta_exit=eta_substrate)

        # The same front stack seen from the substrate. Each layer matrix L has equal
        # diagonal elements, so L^T = J L J with J = [[0, 1], [1, 0]]; the product in
        # reverse order is then (J M J)^T, i.e. M with m11 and m22 swapped.
        r_21, t_21 = _tangential_r_t_from_matrix(eta_substrate, m22, m12, m21, m11, eta_exit=eta_inc)

        # Back face of the substrate: a single interface into the exit medium.
        r_23 = _tangential_r(eta_substrate, eta_exit)
        t_23 = _tangential_t(eta_substrate, eta_exit)

        # One pass through the substrate, and one full round trip (an echo).
        phase = np.exp(
            1j
            * _clamp_imag_phase(
                propagation_phase(omega, layer_n[:, substrate], layer_thickness[substrate], cos_theta[:, substrate])
            )
        )
        round_trip = r_21 * r_23 * phase**2

        # Finite geometric sums: N transmitted passes, N - 1 echoes in reflection.
        passes = int(num_substrate_passes)
        t_tan = t_12 * t_23 * phase * (1.0 - round_trip**passes) / (1.0 - round_trip)
        r_tan = r_12 + t_12 * r_23 * t_21 * phase**2 * (1.0 - round_trip ** (passes - 1)) / (1.0 - round_trip)

    # Power fractions from the power flow normal to the layers; independent of the p sign convention.
    R = np.abs(r_tan) ** 2
    T = np.real(eta_exit) / np.real(eta_inc) * np.abs(t_tan) ** 2
    A = 1.0 - R - T

    # Tangential-field ratios to the textbook r and t (only p changes).
    r = _r_from_tangential(r_tan, polarization)
    t = _t_from_tangential(t_tan, cos_theta[:, 0], cos_theta[:, -1], polarization)

    # The opposite time convention is the complex conjugate.
    if conjugate_model:
        r = np.conjugate(r)
        t = np.conjugate(t)

    return Result(
        omega=omega,
        r=np.asarray(r, dtype=np.complex128),
        t=np.asarray(t, dtype=np.complex128),
        R=np.asarray(R, dtype=np.float64),
        T=np.asarray(T, dtype=np.float64),
        A=np.asarray(A, dtype=np.float64),
    )
