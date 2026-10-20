"""Single interfaces: Snell's law, admittances, Fresnel coefficients and their sign convention."""

import numpy as np
import pytest

from cjctmm import (
    Polarization,
    VACUUM_IMPEDANCE,
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

# Both polarizations, typed as Polarization so type checkers accept them where "s" or "p" is expected.
POLARIZATIONS: tuple[Polarization, ...] = ("s", "p")

# Glass, the standard simple value for hand calculations.
N_GLASS = 1.5


def test_snell_air_to_glass_at_30_degrees():
    """Air to glass at 30 deg: sin(theta_t) = sin(30 deg)/1.5 = 1/3."""
    theta_t = snell_theta(1.0, N_GLASS, np.radians(30.0))
    assert np.isclose(np.sin(theta_t), 1.0 / 3.0)
    # A real angle below the critical angle.
    assert abs(theta_t.imag) < 1e-15
    assert np.isclose(snell_cos_theta(1.0, N_GLASS, np.radians(30.0)), np.sqrt(8.0 / 9.0))


def test_critical_angle_glass_to_air_and_total_internal_reflection():
    """Glass to air: critical angle 41.81 deg; beyond it cos(theta) is imaginary and R = 1."""
    theta_c = critical_angle(N_GLASS, 1.0)
    assert np.isclose(np.degrees(theta_c), 41.81, atol=0.005)

    # Just beyond the critical angle the transmitted wave is evanescent.
    theta = theta_c + np.radians(5.0)
    cos_t = snell_cos_theta(N_GLASS, 1.0, theta)
    assert abs(cos_t.real) < 1e-15 and cos_t.imag > 0.0
    for polarization in POLARIZATIONS:
        assert np.isclose(fresnel_R(N_GLASS, 1.0, theta, polarization), 1.0)
        assert np.isclose(fresnel_T(N_GLASS, 1.0, theta, polarization), 0.0)

    # From the optically thinner side there is no critical angle.
    assert np.isnan(critical_angle(1.0, N_GLASS))


def test_snell_theta_beyond_critical_angle():
    """Beyond the critical angle theta = pi/2 - i*arccosh(sin(theta))."""
    theta_inc = np.radians(60.0)
    sin_t = N_GLASS * np.sin(theta_inc)
    theta_t = snell_theta(N_GLASS, 1.0, theta_inc)
    assert np.isclose(theta_t, np.pi / 2 - 1j * np.arccosh(sin_t))
    # Consistent with the decaying-branch cosine.
    assert np.isclose(np.cos(theta_t), snell_cos_theta(N_GLASS, 1.0, theta_inc))


def test_snell_cos_theta_decays_in_absorbing_medium():
    """In an absorbing medium cos(theta) and n*cos(theta) both have Im >= 0 (decaying wave)."""
    n = np.array([2.0 + 0.5j, 0.2 + 3.0j, 1.0 + 1e-6j])
    for theta in np.radians([0.0, 30.0, 60.0, 89.0]):
        cos_t = snell_cos_theta(1.0, n, theta)
        assert np.all(cos_t.imag >= 0.0)
        assert np.all((n * cos_t).imag >= 0.0)


def test_snell_cos_theta_ignores_rounding_noise_in_the_invariant():
    """A propagating wave keeps Re(cos) > 0 when the Snell invariant carries rounding noise."""
    # The complex angle in an absorbing film gives n_film*sin(theta_film) = sin(40 deg) only up to
    # ~1e-16; a naive Im >= 0 rule would flip the cosine in the glass behind it to about -0.90.
    theta_film = snell_theta(1.0, 2.0 + 0.1j, np.radians(40.0))
    cos_glass = snell_cos_theta(2.0 + 0.1j, N_GLASS, theta_film)
    assert np.isclose(cos_glass, snell_cos_theta(1.0, N_GLASS, np.radians(40.0)))
    assert cos_glass.real > 0.0


def test_admittance_at_normal_incidence_is_n_over_z0():
    """At normal incidence both polarizations have eta = n/Z0; oblique, s and p tilt opposite ways."""
    assert np.isclose(admittance(N_GLASS, 1.0, "s"), N_GLASS / VACUUM_IMPEDANCE)
    assert np.isclose(admittance(N_GLASS, 1.0, "p"), N_GLASS / VACUUM_IMPEDANCE)
    assert np.isclose(admittance(1.0, 0.5, "s"), 0.5 / VACUUM_IMPEDANCE)
    assert np.isclose(admittance(1.0, 0.5, "p"), 2.0 / VACUUM_IMPEDANCE)


def test_fresnel_air_to_glass_at_normal_incidence():
    """Air to glass at normal incidence: r_s = -0.2, r_p = +0.2, t = 0.8, R = 0.04, T = 0.96."""
    assert np.isclose(fresnel_r(1.0, N_GLASS, 0.0, "s"), -0.2)
    assert np.isclose(fresnel_r(1.0, N_GLASS, 0.0, "p"), +0.2)
    for polarization in POLARIZATIONS:
        assert np.isclose(fresnel_t(1.0, N_GLASS, 0.0, polarization), 0.8)
        assert np.isclose(fresnel_R(1.0, N_GLASS, 0.0, polarization), 0.04)
        assert np.isclose(fresnel_T(1.0, N_GLASS, 0.0, polarization), 0.96)


def test_fresnel_at_grazing_incidence():
    """Towards grazing incidence both r_s and r_p go to -1 (total reflection, phase flip)."""
    theta = np.pi / 2 - 1e-7
    assert np.isclose(fresnel_r(1.0, N_GLASS, theta, "s"), -1.0, atol=1e-6)
    assert np.isclose(fresnel_r(1.0, N_GLASS, theta, "p"), -1.0, atol=1e-6)


def test_brewster_angle_zeroes_r_p_and_flips_its_sign():
    """At arctan(1.5) = 56.31 deg r_p = 0; below it r_p > 0, above it r_p < 0."""
    theta_b = brewster_angle(1.0, N_GLASS)
    assert np.isclose(np.degrees(theta_b), 56.31, atol=0.005)
    assert abs(fresnel_r(1.0, N_GLASS, theta_b, "p")) < 1e-15
    assert fresnel_r(1.0, N_GLASS, theta_b - 0.01, "p").real > 0.0
    assert fresnel_r(1.0, N_GLASS, theta_b + 0.01, "p").real < 0.0
    # r_s does not vanish there.
    assert fresnel_R(1.0, N_GLASS, theta_b, "s") > 0.1


def test_fresnel_textbook_closed_forms():
    """r and t equal the textbook (Hecht) formulas in n and cos(theta) at oblique incidence."""
    n_1, n_2, theta = 1.0, N_GLASS, np.radians(40.0)
    cos_1 = np.cos(theta)
    cos_2 = np.sqrt(1.0 - (n_1 * np.sin(theta) / n_2) ** 2)
    assert np.isclose(fresnel_r(n_1, n_2, theta, "s"), (n_1 * cos_1 - n_2 * cos_2) / (n_1 * cos_1 + n_2 * cos_2))
    assert np.isclose(fresnel_r(n_1, n_2, theta, "p"), (n_2 * cos_1 - n_1 * cos_2) / (n_2 * cos_1 + n_1 * cos_2))
    assert np.isclose(fresnel_t(n_1, n_2, theta, "s"), 2 * n_1 * cos_1 / (n_1 * cos_1 + n_2 * cos_2))
    assert np.isclose(fresnel_t(n_1, n_2, theta, "p"), 2 * n_1 * cos_1 / (n_2 * cos_1 + n_1 * cos_2))


@pytest.mark.parametrize("polarization", POLARIZATIONS)
def test_fresnel_energy_conservation_lossless(polarization: Polarization):
    """R + T = 1 at a lossless interface, from both sides, below the critical angle."""
    theta = np.radians(np.linspace(0.0, 85.0, 18))
    assert np.allclose(fresnel_R(1.0, N_GLASS, theta, polarization) + fresnel_T(1.0, N_GLASS, theta, polarization), 1.0)
    theta = np.radians(np.linspace(0.0, 40.0, 9))
    assert np.allclose(fresnel_R(N_GLASS, 1.0, theta, polarization) + fresnel_T(N_GLASS, 1.0, theta, polarization), 1.0)


def test_fresnel_rejects_unknown_polarization():
    """Anything but 's' or 'p' (e.g. 'P' or the old integer 0) raises a clear error."""
    # The wrong types are the point of this test, so the type-checker warnings are silenced.
    with pytest.raises(ValueError, match="polarization must be 's'"):
        # noinspection PyTypeChecker
        fresnel_r(1.0, N_GLASS, 0.0, "P")
    with pytest.raises(ValueError, match="polarization must be 's'"):
        # noinspection PyTypeChecker
        fresnel_t(1.0, N_GLASS, 0.0, 0)


def boundary_condition_coefficients(n_1, n_2, theta_inc, polarization):
    """
    r and t of one interface from Maxwell's boundary conditions, built from scratch.

    Plane of incidence xz, interface z = 0, medium 1 at z < 0, y normal to the
    plane of incidence. Each wave has a direction k_hat (complex in an absorbing
    medium, with k_hat . k_hat = 1) and fields related by H = k_hat x E / Z and
    E = Z H x k_hat, with Z = Z0/n. For p, H of all three waves points along +y;
    for s, E does. r and t are ratios of the electric-field amplitudes along
    these reference directions.
    """
    # Snell's law and the decaying branch, written out independently of the package.
    sin_1, cos_1 = np.sin(theta_inc), np.cos(theta_inc)
    sin_2 = n_1 * sin_1 / n_2
    cos_2 = np.sqrt(1.0 - sin_2**2 + 0j)
    if cos_2.imag < 0.0:
        cos_2 = -cos_2

    # Wave directions: incident, reflected (k_z flipped) and transmitted.
    k_inc = np.array([sin_1, 0.0, cos_1], dtype=complex)
    k_ref = np.array([sin_1, 0.0, -cos_1], dtype=complex)
    k_tra = np.array([sin_2, 0.0, cos_2], dtype=complex)
    y_hat = np.array([0.0, 1.0, 0.0], dtype=complex)
    z_1, z_2 = VACUUM_IMPEDANCE / n_1, VACUUM_IMPEDANCE / n_2

    def fields(k_hat, z):
        """E and H of a wave with unit electric-field amplitude along its reference direction."""
        if polarization == "p":
            h = y_hat / z  # |E| = Z |H| = 1
            return z * np.cross(h, k_hat), h
        e = y_hat
        return e, np.cross(k_hat, e) / z

    # Tangential components (x, y) of E and H for each wave.
    def tangential(k_hat, z):
        e, h = fields(k_hat, z)
        return np.array([e[0], e[1], h[0], h[1]])

    inc, ref, tra = tangential(k_inc, z_1), tangential(k_ref, z_1), tangential(k_tra, z_2)

    # Continuity: inc + r*ref = t*tra. Only two of the four rows are non-trivial:
    # (E_x, H_y) for p and (E_y, H_x) for s; the other two read 0 = 0.
    rows = [0, 3] if polarization == "p" else [1, 2]
    matrix = np.column_stack([ref[rows], -tra[rows]])
    r, t = np.linalg.solve(matrix, -inc[rows])
    return r, t


@pytest.mark.parametrize("polarization", POLARIZATIONS)
@pytest.mark.parametrize(
    "n_1, n_2",
    [(1.0, 1.5), (1.5, 1.0), (1.0, 3.4 + 0.2j), (1.0, 0.3 + 4.0j), (1.3 + 0.1j, 2.0 + 0.5j)],
)
def test_fresnel_matches_boundary_conditions(polarization: Polarization, n_1, n_2):
    """The package's r and t equal a direct solve of tangential E and H continuity, at many angles."""
    for theta in np.radians([0.0, 20.0, 45.0, 60.0, 80.0]):
        r, t = boundary_condition_coefficients(n_1, n_2, theta, polarization)
        assert np.isclose(fresnel_r(n_1, n_2, theta, polarization), r, rtol=1e-12, atol=1e-14)
        assert np.isclose(fresnel_t(n_1, n_2, theta, polarization), t, rtol=1e-12, atol=1e-14)


def test_boundary_condition_construction_pins_r_p_sign():
    """The from-scratch construction itself gives r_p = +0.2 for air to glass at normal incidence."""
    # Guards the test helper: if its field directions changed, this would catch it.
    r_p, t_p = boundary_condition_coefficients(1.0, N_GLASS, 0.0, "p")
    assert np.isclose(r_p, 0.2) and np.isclose(t_p, 0.8)
