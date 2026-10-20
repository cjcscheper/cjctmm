"""solve_tmm interface: shapes, validation, conjugate_model, the frozen Result, and finite substrate passes."""

import dataclasses
import warnings

import numpy as np
import pytest

from cjctmm import SPEED_OF_LIGHT, Polarization, Result, fresnel_r, fresnel_t, propagation_phase, solve_tmm

# Both polarizations, typed as Polarization so type checkers accept them where "s" or "p" is expected.
POLARIZATIONS: tuple[Polarization, ...] = ("s", "p")

NAN = np.nan

# Five frequencies around 1 THz and a simple film on a substrate.
OMEGA = 2 * np.pi * np.linspace(0.5e12, 1.5e12, 5)
THICKNESS = [NAN, 1e-6, NAN]
N_LAYERS = [1.0, 2.0 + 0.1j, 3.4]


def test_result_shapes_and_types():
    """Every field of Result has shape (M,), with complex r, t and real R, T, A."""
    result = solve_tmm(OMEGA, THICKNESS, N_LAYERS)
    assert isinstance(result, Result)
    for name in ("omega", "r", "t", "R", "T", "A"):
        assert getattr(result, name).shape == OMEGA.shape
    assert result.r.dtype == np.complex128 and result.t.dtype == np.complex128
    assert result.R.dtype == np.float64 and result.T.dtype == np.float64 and result.A.dtype == np.float64
    assert np.array_equal(result.omega, OMEGA)


def test_frequency_independent_indices_broadcast():
    """layer_n of shape (N,) gives the same result as the same indices repeated for every frequency."""
    full = np.tile(np.asarray(N_LAYERS, dtype=complex), (OMEGA.size, 1))
    assert np.array_equal(solve_tmm(OMEGA, THICKNESS, N_LAYERS).r, solve_tmm(OMEGA, THICKNESS, full).r)


def test_scalar_omega_is_accepted():
    """A single frequency may be given as a scalar; the result then has shape (1,)."""
    assert solve_tmm(OMEGA[0], THICKNESS, N_LAYERS).r.shape == (1,)


def test_result_is_frozen():
    """Result fields cannot be reassigned."""
    result = solve_tmm(OMEGA, THICKNESS, N_LAYERS)
    # setattr instead of result.R = ...: the write is meant to fail, and this way the IDE does not flag it.
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, "R", np.zeros(5))


def test_conjugate_model_conjugates_r_and_t_only():
    """conjugate_model=True returns conj(r), conj(t) and leaves the power fractions unchanged."""
    plain = solve_tmm(OMEGA, THICKNESS, N_LAYERS, theta_inc=0.3, polarization="p")
    conjugated = solve_tmm(OMEGA, THICKNESS, N_LAYERS, theta_inc=0.3, polarization="p", conjugate_model=True)
    assert np.array_equal(conjugated.r, np.conjugate(plain.r))
    assert np.array_equal(conjugated.t, np.conjugate(plain.t))
    for name in ("R", "T", "A"):
        assert np.array_equal(getattr(conjugated, name), getattr(plain, name))


def test_propagation_phase_value():
    """delta = (omega/c) n d cos(theta): a quarter-wave layer at normal incidence has delta = pi/2."""
    wavelength, n = 1e-6, 2.0
    omega = 2 * np.pi * SPEED_OF_LIGHT / wavelength
    assert np.isclose(propagation_phase(omega, n, wavelength / (4 * n), 1.0), np.pi / 2)
    # Absorption: |exp(i delta)| = exp(-Im delta).
    assert np.isclose(propagation_phase(omega, 2.0 + 0.1j, 1e-6, 1.0).imag, omega / SPEED_OF_LIGHT * 0.1 * 1e-6)


def test_polarization_is_validated():
    """Only 's' and 'p' are accepted, with a message naming both."""
    # The wrong types are the point of this test, so the type-checker warnings are silenced.
    with pytest.raises(ValueError, match=r"polarization must be 's' \(senkrecht, TE\) or 'p'"):
        # noinspection PyTypeChecker
        solve_tmm(OMEGA, THICKNESS, N_LAYERS, polarization="TE")
    with pytest.raises(ValueError, match="polarization must be"):
        # noinspection PyTypeChecker
        solve_tmm(OMEGA, THICKNESS, N_LAYERS, polarization=1)


def test_wrong_shapes_raise_clear_errors():
    """Wrong array shapes name the argument and the expected shape."""
    with pytest.raises(ValueError, match="omega must be a 1D array"):
        solve_tmm(OMEGA.reshape(5, 1), THICKNESS, N_LAYERS)
    with pytest.raises(ValueError, match="layer_thickness must be a 1D array"):
        solve_tmm(OMEGA, [NAN], [1.0])
    with pytest.raises(ValueError, match=r"layer_n must have shape \(len\(omega\), len\(layer_thickness\)\) = \(5, 3\)"):
        solve_tmm(OMEGA, THICKNESS, np.ones((4, 3)))
    with pytest.raises(ValueError, match="layer_n must have shape"):
        solve_tmm(OMEGA, THICKNESS, [1.0, 2.0])


def test_invalid_values_raise_clear_errors():
    """Non-positive frequencies, bad angles and bad thicknesses are rejected."""
    with pytest.raises(ValueError, match="omega must be positive"):
        solve_tmm(np.array([0.0, 1.0]), THICKNESS, N_LAYERS)
    with pytest.raises(ValueError, match=r"theta_inc must be in \[0, pi/2\)"):
        solve_tmm(OMEGA, THICKNESS, N_LAYERS, theta_inc=np.pi / 2)
    with pytest.raises(ValueError, match="must be NaN"):
        solve_tmm(OMEGA, [0.0, 1e-6, NAN], N_LAYERS)
    with pytest.raises(ValueError, match="finite and >= 0, got -1e-06 at index 1"):
        solve_tmm(OMEGA, [NAN, -1e-6, NAN], N_LAYERS)
    with pytest.raises(ValueError, match="finite and >= 0, got nan at index 1"):
        solve_tmm(OMEGA, [NAN, NAN, NAN], N_LAYERS)


# --- Finite substrate passes -------------------------------------------------------


def test_substrate_passes_value_errors():
    """Too few layers, a NaN substrate thickness and fewer than one pass are rejected."""
    with pytest.raises(ValueError, match="at least 3 layers"):
        solve_tmm(OMEGA, [NAN, NAN], [1.0, 3.4], num_substrate_passes=2)
    with pytest.raises(ValueError, match="finite substrate thickness"):
        solve_tmm(OMEGA, [NAN, NAN, NAN], N_LAYERS, num_substrate_passes=2)
    with pytest.raises(ValueError, match="num_substrate_passes must be >= 1"):
        solve_tmm(OMEGA, THICKNESS, N_LAYERS, num_substrate_passes=0)


def test_single_pass_is_the_direct_pulse():
    """One pass through a bare substrate: t = t12 t23 exp(i delta) and r = r12, without echoes."""
    n_sub, d = 3.4, 500e-6
    result = solve_tmm(OMEGA, [NAN, d, NAN], [1.0, n_sub, 1.0], num_substrate_passes=1)
    phase = np.exp(1j * OMEGA / SPEED_OF_LIGHT * n_sub * d)
    assert np.allclose(result.t, fresnel_t(1.0, n_sub, 0.0) * fresnel_t(n_sub, 1.0, 0.0) * phase)
    assert np.allclose(result.r, fresnel_r(1.0, n_sub, 0.0))


@pytest.mark.parametrize("polarization", POLARIZATIONS)
def test_many_passes_converge_to_coherent_result(polarization: Polarization):
    """For an absorbing substrate the echoes die out, so many passes equal the fully coherent result."""
    thickness = [NAN, 200e-9, 500e-6, NAN]
    n = [1.0, 2.0 + 0.5j, 3.4 + 0.05j, 1.0]
    coherent = solve_tmm(OMEGA, thickness, n, polarization=polarization, theta_inc=0.4)
    gated = solve_tmm(OMEGA, thickness, n, polarization=polarization, theta_inc=0.4, num_substrate_passes=200)
    assert np.allclose(gated.r, coherent.r, rtol=1e-10) and np.allclose(gated.t, coherent.t, rtol=1e-10)
    assert np.allclose(gated.T, coherent.T, rtol=1e-10)


def test_thin_film_on_thick_substrate_does_not_warn():
    """A thin front film has all its echoes long before the gate: no warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solve_tmm(OMEGA, [NAN, 100e-9, 500e-6, NAN], [1.0, 2.0, 3.4, 1.0], num_substrate_passes=1)


def test_front_layer_as_thick_as_substrate_warns():
    """A front layer about as thick as the substrate has echoes beyond the gate: a UserWarning names it."""
    # Round trips: front 2*2.0*1 mm, substrate 2*3.4*1 mm; with one pass the first
    # front echo beyond the gate is the 2nd, |rho|^2/(1-|rho|) ~ 8e-3 > 1e-3.
    with pytest.warns(UserWarning, match="front layer 1 has echoes later than the substrate gate"):
        solve_tmm(OMEGA, [NAN, 1e-3, 1e-3, NAN], [1.0, 2.0, 3.4, 1.0], num_substrate_passes=1)
    # A looser tolerance accepts the same stack.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solve_tmm(
            OMEGA, [NAN, 1e-3, 1e-3, NAN], [1.0, 2.0, 3.4, 1.0], num_substrate_passes=1, substrate_pass_tolerance=0.1
        )
