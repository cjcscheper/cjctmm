"""solve_tmm against analytic results: Fresnel, Airy, coatings, Fabry-Perot, Bragg mirror, thin conducting film."""

import numpy as np
import pytest

from cjctmm import (
    Polarization,
    SPEED_OF_LIGHT,
    VACUUM_IMPEDANCE,
    VACUUM_PERMITTIVITY,
    fresnel_r,
    fresnel_t,
    snell_cos_theta,
    snell_theta,
    solve_tmm,
)

# Both polarizations, typed as Polarization so type checkers accept them where "s" or "p" is expected.
POLARIZATIONS: tuple[Polarization, ...] = ("s", "p")

NAN = np.nan
N_GLASS = 1.5

# Design wavelength for the visible-light tests; the method itself has no scale.
WAVELENGTH = 550e-9
OMEGA = np.array([2 * np.pi * SPEED_OF_LIGHT / WAVELENGTH])


@pytest.mark.parametrize("polarization", POLARIZATIONS)
@pytest.mark.parametrize("n_exit", [N_GLASS, 3.4 + 0.2j, 0.3 + 4.0j])
def test_single_interface_equals_fresnel(polarization: Polarization, n_exit):
    """A stack of just two semi-infinite media gives exactly the Fresnel coefficients."""
    for theta in np.radians([0.0, 30.0, 70.0]):
        result = solve_tmm(OMEGA, [NAN, NAN], [1.0, n_exit], polarization=polarization, theta_inc=theta)
        assert np.allclose(result.r, fresnel_r(1.0, n_exit, theta, polarization), rtol=1e-14, atol=0)
        assert np.allclose(result.t, fresnel_t(1.0, n_exit, theta, polarization), rtol=1e-14, atol=0)


def test_total_internal_reflection_in_a_stack():
    """Glass into air beyond the critical angle: R = 1 and T = 0, also through a thin film."""
    theta = np.radians(50.0)
    for polarization in POLARIZATIONS:
        result = solve_tmm(OMEGA, [NAN, 100e-9, NAN], [N_GLASS, 1.3, 1.0], polarization=polarization, theta_inc=theta)
        assert np.isclose(result.R[0], 1.0) and np.isclose(result.T[0], 0.0, atol=1e-15)


@pytest.mark.parametrize("polarization", POLARIZATIONS)
def test_lossless_stack_conserves_energy(polarization: Polarization):
    """R + T = 1 for a lossless multilayer at many frequencies and angles."""
    omega = np.linspace(0.5, 2.0, 50) * OMEGA[0]
    thickness = [NAN, 80e-9, 230e-9, 1.1e-6, NAN]
    n = [1.0, 2.3, 1.38, 1.9, N_GLASS]
    for theta in np.radians([0.0, 25.0, 50.0, 75.0]):
        result = solve_tmm(omega, thickness, n, polarization=polarization, theta_inc=theta)
        assert np.allclose(result.R + result.T, 1.0, rtol=0, atol=1e-13)
        assert np.allclose(result.A, 0.0, atol=1e-13)


def test_absorbing_stack_has_non_negative_absorptance():
    """With absorbing layers A = 1 - R - T is positive and R, T lie within [0, 1]."""
    omega = np.linspace(0.5, 2.0, 50) * OMEGA[0]
    thickness = [NAN, 30e-9, 200e-9, NAN]
    n = [1.0, 0.2 + 3.0j, 2.0 + 0.1j, N_GLASS]
    for polarization in POLARIZATIONS:
        for theta in np.radians([0.0, 40.0, 80.0]):
            result = solve_tmm(omega, thickness, n, polarization=polarization, theta_inc=theta)
            assert np.all(result.A > 0.0)
            assert np.all((result.R >= 0) & (result.R <= 1) & (result.T >= 0) & (result.T <= 1))


def test_normal_incidence_s_and_p_agree():
    """At normal incidence s and p are the same wave: equal R, T, A and t, and r_p = -r_s."""
    omega = np.linspace(0.5, 2.0, 20) * OMEGA[0]
    thickness = [NAN, 30e-9, 200e-9, NAN]
    n = [1.0, 0.2 + 3.0j, 2.0 + 0.1j, N_GLASS]
    s = solve_tmm(omega, thickness, n, polarization="s")
    p = solve_tmm(omega, thickness, n, polarization="p")
    for name in ("R", "T", "A", "t"):
        assert np.allclose(getattr(s, name), getattr(p, name), rtol=1e-14, atol=0)
    # The textbook p convention points the reflected E reference opposite to s at normal incidence.
    assert np.allclose(p.r, -s.r, rtol=1e-14, atol=0)


@pytest.mark.parametrize("polarization", POLARIZATIONS)
def test_zero_thickness_layer_has_no_effect(polarization: Polarization):
    """A layer of zero thickness, whatever its index, changes nothing."""
    omega = np.linspace(0.5, 2.0, 20) * OMEGA[0]
    theta = np.radians(35.0)
    bare = solve_tmm(omega, [NAN, 200e-9, NAN], [1.0, 2.0, N_GLASS], polarization=polarization, theta_inc=theta)
    extra = solve_tmm(
        omega, [NAN, 0.0, 200e-9, NAN], [1.0, 3.0 + 1.0j, 2.0, N_GLASS], polarization=polarization, theta_inc=theta
    )
    assert np.allclose(extra.r, bare.r, rtol=1e-13) and np.allclose(extra.t, bare.t, rtol=1e-13)


@pytest.mark.parametrize("polarization", POLARIZATIONS)
def test_index_matched_layers(polarization: Polarization):
    """Splitting a layer into two of the same index changes nothing; an incident-matched layer only adds phase."""
    omega = np.linspace(0.5, 2.0, 20) * OMEGA[0]
    theta = np.radians(35.0)
    n_film = 2.0 + 0.05j
    bare = solve_tmm(omega, [NAN, 300e-9, NAN], [1.0, n_film, N_GLASS], polarization=polarization, theta_inc=theta)

    # Two layers of the same material act as one layer of the summed thickness.
    split = solve_tmm(
        omega, [NAN, 100e-9, 200e-9, NAN], [1.0, n_film, n_film, N_GLASS], polarization=polarization, theta_inc=theta
    )
    assert np.allclose(split.r, bare.r, rtol=1e-13) and np.allclose(split.t, bare.t, rtol=1e-13)

    # A layer of the incident medium in front only moves the reference plane:
    # r gains the round-trip phase exp(2i delta), t the single-pass phase exp(i delta).
    d_air = 400e-9
    shifted = solve_tmm(
        omega, [NAN, d_air, 300e-9, NAN], [1.0, 1.0, n_film, N_GLASS], polarization=polarization, theta_inc=theta
    )
    delta = omega / SPEED_OF_LIGHT * d_air * np.cos(theta)
    assert np.allclose(shifted.r, bare.r * np.exp(2j * delta), rtol=1e-13)
    assert np.allclose(shifted.t, bare.t * np.exp(1j * delta), rtol=1e-13)
    assert np.allclose(shifted.R, bare.R, rtol=1e-13)


@pytest.mark.parametrize("polarization", POLARIZATIONS)
def test_thin_film_equals_airy_formula(polarization: Polarization):
    """One absorbing film at oblique incidence equals the Airy sum built from Fresnel coefficients."""
    omega = np.linspace(0.5, 2.0, 40) * OMEGA[0]
    n_1, n_2, n_3 = 1.0, 2.0 + 0.1j, N_GLASS
    d = 300e-9
    theta = np.radians(40.0)
    result = solve_tmm(omega, [NAN, d, NAN], [n_1, n_2, n_3], polarization=polarization, theta_inc=theta)

    # Fresnel coefficients at both faces; the angle in the film from Snell's law.
    r_12 = fresnel_r(n_1, n_2, theta, polarization)
    t_12 = fresnel_t(n_1, n_2, theta, polarization)
    cos_2 = snell_cos_theta(n_1, n_2, theta)
    theta_2 = snell_theta(n_1, n_2, theta)
    r_23 = fresnel_r(n_2, n_3, theta_2, polarization)
    t_23 = fresnel_t(n_2, n_3, theta_2, polarization)
    # The complex angle in the absorbing film must reproduce its own cosine.
    assert np.isclose(np.cos(theta_2), cos_2)

    # Airy: all round trips in the film summed as a geometric series.
    phase = np.exp(1j * omega / SPEED_OF_LIGHT * n_2 * d * cos_2)
    r_airy = (r_12 + r_23 * phase**2) / (1 + r_12 * r_23 * phase**2)
    t_airy = t_12 * t_23 * phase / (1 + r_12 * r_23 * phase**2)
    assert np.allclose(result.r, r_airy, rtol=1e-12)
    assert np.allclose(result.t, t_airy, rtol=1e-12)


def test_quarter_wave_antireflection_coating():
    """A quarter-wave layer of index sqrt(1.5) on glass reflects nothing at its design wavelength."""
    n_coating = np.sqrt(N_GLASS)
    d = WAVELENGTH / (4 * n_coating)
    result = solve_tmm(OMEGA, [NAN, d, NAN], [1.0, n_coating, N_GLASS])
    assert result.R[0] < 1e-30
    assert np.isclose(result.T[0], 1.0)


def test_half_wave_layer_is_absent():
    """A half-wave layer at its design wavelength gives the bare-substrate reflection."""
    n_layer = 2.0
    d = WAVELENGTH / (2 * n_layer)
    result = solve_tmm(OMEGA, [NAN, d, NAN], [1.0, n_layer, N_GLASS])
    assert np.isclose(result.R[0], 0.04)
    assert np.isclose(result.r[0], -0.2)


@pytest.mark.parametrize("theta_deg", [0.0, 30.0])
def test_fabry_perot_slab_transmits_fully_at_resonance(theta_deg):
    """A freestanding lossless slab has T = 1 where 2 n d cos(theta) = m * wavelength."""
    n_slab, d = 3.4, 500e-6
    theta = np.radians(theta_deg)
    cos_slab = np.sqrt(1 - (np.sin(theta) / n_slab) ** 2)
    m = np.arange(1, 6)
    omega_resonance = 2 * np.pi * m * SPEED_OF_LIGHT / (2 * n_slab * d * cos_slab)
    for polarization in POLARIZATIONS:
        result = solve_tmm(omega_resonance, [NAN, d, NAN], [1.0, n_slab, 1.0], polarization=polarization, theta_inc=theta)
        assert np.allclose(result.T, 1.0)

        # Halfway between resonances the slab reflects its maximum, 1 - T_min = 4R/(1+R)^2 with R of one face.
        between = solve_tmm(
            omega_resonance + np.pi * SPEED_OF_LIGHT / (2 * n_slab * d * cos_slab),
            [NAN, d, NAN],
            [1.0, n_slab, 1.0],
            polarization=polarization,
            theta_inc=theta,
        )
        r_face = abs(fresnel_r(1.0, n_slab, theta, polarization)) ** 2
        assert np.allclose(between.R, 4 * r_face / (1 + r_face) ** 2)


@pytest.mark.parametrize("num_pairs", [1, 4, 10])
def test_bragg_mirror_matches_closed_form(num_pairs):
    """A quarter-wave stack (H L)^N on glass reflects R = ((n0 - Y)/(n0 + Y))^2 with Y = (n_H/n_L)^(2N) n_s."""
    n_high, n_low, n_sub = 2.3, 1.38, 1.52
    thickness = [NAN] + [WAVELENGTH / (4 * n_high), WAVELENGTH / (4 * n_low)] * num_pairs + [NAN]
    n = [1.0] + [n_high, n_low] * num_pairs + [n_sub]
    result = solve_tmm(OMEGA, thickness, n)

    # Each quarter-wave layer maps the admittance behind it to n_layer^2 / Y.
    y = (n_high / n_low) ** (2 * num_pairs) * n_sub
    assert np.isclose(result.R[0], ((1.0 - y) / (1.0 + y)) ** 2, rtol=1e-12)


def test_thin_conducting_film_tinkham_formula():
    """A conducting film much thinner than the wavelength transmits t/t_sub = (1 + n_s)/(1 + n_s + Z0 sigma d)."""
    omega = 2 * np.pi * np.linspace(0.1e12, 3e12, 30)
    n_sub, sigma, d = 3.4, 5e6, 2e-9  # Z0*sigma*d ~ 3.8: strong but not total screening
    # Film index from its permittivity: background 1 plus the conductivity term.
    n_film = np.sqrt(1.0 + 1j * sigma / (VACUUM_PERMITTIVITY * omega))
    layer_n = np.column_stack([np.ones_like(omega), n_film, np.full_like(omega, n_sub)])

    film = solve_tmm(omega, [NAN, d, NAN], layer_n)
    bare = solve_tmm(omega, [NAN, NAN], [1.0, n_sub])
    tinkham = (1 + n_sub) / (1 + n_sub + VACUUM_IMPEDANCE * sigma * d)
    # Thin-film limit: the neglected terms are of order (omega d/c) * n_s ~ 1e-4 here.
    assert np.allclose(film.t / bare.t, tinkham, rtol=1e-3)


def test_opaque_layer_stays_finite():
    """A thick metal layer (Im(delta) far above the clamp) gives finite r, t and T = 0."""
    result = solve_tmm(OMEGA, [NAN, 10e-6, NAN], [1.0, 0.1 + 5.0j, N_GLASS])
    assert np.all(np.isfinite(result.r)) and np.all(np.isfinite(result.t))
    assert result.T[0] < 1e-30
    # Equal to the reflection of a semi-infinite block of the metal.
    assert np.isclose(result.r[0], fresnel_r(1.0, 0.1 + 5.0j, 0.0))
