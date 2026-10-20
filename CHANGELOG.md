# Changelog

All notable changes to cjctmm. Versions follow MAJOR.MINOR.PATCH: patch for
fixes, minor for new features, major for changes that break existing scripts.

## [Unreleased]

### Added
- `solve_tmm(omega, layer_thickness, layer_n, ...)`: transfer-matrix method for
  planar multilayer stacks, s and p polarization at oblique incidence, absorbing
  layers and evanescent waves, returning a frozen `Result` (`omega, r, t, R, T, A`).
- `num_substrate_passes` for time-gated measurements, with a `UserWarning` when
  front-layer echoes beyond the implied gate exceed `substrate_pass_tolerance`.
- `conjugate_model` for the `exp(+i omega t)` time convention.
- Single-interface functions: `fresnel_r`, `fresnel_t`, `fresnel_R`, `fresnel_T`,
  `critical_angle`, `brewster_angle`, `admittance`, `snell_cos_theta`,
  `snell_theta`, and `propagation_phase`.
- Physical constants `SPEED_OF_LIGHT`, `VACUUM_PERMEABILITY`,
  `VACUUM_PERMITTIVITY`, `VACUUM_IMPEDANCE`.
- THEORY.md (derivation, conventions, substrate-pass validity), examples
  (antireflection coating, Fabry–Pérot slab, angle dependence, thin conducting
  film), tests and a GitHub Actions workflow.
- `examples` extra installing cjcplot pinned to `v0.2.0`; the examples plot
  with cjcplot. `pip install cjctmm` itself stays numpy-only.
- `requirements.txt` with the dependencies of `pyproject.toml`, and a test that
  keeps them in sync.
