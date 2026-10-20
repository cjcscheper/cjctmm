# AGENTS.md — cjctmm

Guidance for AI coding agents (and contributors) working in this repository.

This file covers much of the same ground as `README.md`, but for a different
reader: `README.md` is for people who *use* cjctmm (install, quick start,
conventions); this file is for agents who *work on* it (layout, design rules,
conventions, tests, releases). `THEORY.md` holds the physics and derivations
for both. When something changes that more than one of them describes, such as
the public API or a convention, update all of them.

cjctmm computes the optical response of a planar multilayer stack with the
transfer-matrix method: arrays in (frequencies, thicknesses, complex indices),
a frozen `Result` out. Material models, layer/stack objects and file loaders
deliberately live elsewhere (a separate materials package imports cjctmm).

## Commands

```bash
pip install -e ".[test]"                         # editable install with pytest
pytest                                           # run all tests
pip install -e ".[examples]"                     # cjcplot, pinned, for the examples
python examples/example_antireflection_coating.py  # each example saves its PNG next to itself
```

## Public API

Scripts import only from the package root, never from submodules:

```python
from cjctmm import solve_tmm, Result, propagation_phase, MAX_IMAG_PHASE
from cjctmm import admittance, snell_cos_theta, snell_theta
from cjctmm import fresnel_r, fresnel_t, fresnel_R, fresnel_T, critical_angle, brewster_angle
from cjctmm import SPEED_OF_LIGHT, VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY, VACUUM_IMPEDANCE
```

## Package layout

```
src/cjctmm/
  __init__.py     public API and __version__ (its only source)
  _constants.py   c, mu0, eps0, Z0 written out (no scipy)
  interfaces.py   Snell's law, admittances, Fresnel coefficients; the private
                  tangential-field helpers shared with tmm.py
  tmm.py          propagation_phase, solve_tmm, Result; characteristic-matrix
                  product, finite substrate passes and their validity check
tests/
  test_interfaces.py    Snell, Fresnel values, p sign pinned from boundary conditions
  test_tmm_analytic.py  solve_tmm against analytic results
  test_tmm_api.py       shapes, validation, Result, substrate passes
examples/               one script per physical situation (plotted with cjcplot), PNG next to it
THEORY.md               derivation, conventions, substrate-pass validity
```

## Physics rules

- **One place per piece of physics.** Admittances, Snell's law, the
  interface coefficients and the tangential-to-full-field conversion live in
  `interfaces.py`; `tmm.py` reuses them. The phase thickness is computed only
  by `propagation_phase`. Do not write a second inline copy of a formula.
- **Characteristic-matrix (admittance) formulation** with the vectorised
  pairwise product. Do not rewrite it into interface and propagation
  matrices: those contain `exp(-i delta)` and overflow sooner in lossy layers.
  Keep the documented clamp `|Im delta| <= MAX_IMAG_PHASE`.
- **Conventions are fixed:** time dependence `exp(-i omega t)`, `n = n' + i kappa`
  with `kappa > 0` absorbing; p sign with `r_p = r_s` at normal incidence
  (`r_p = -0.2` air to glass), the opposite of Hecht / Born & Wolf (THEORY.md
  compares both). There is no switch for the p sign. `test_fresnel_matches_boundary_conditions` pins it from Maxwell's
  boundary conditions; never change that test to match code.
- **Snell's law** uses the incident medium's invariant `n_0 sin(theta_0)` in
  every layer and returns `cos(theta)` directly, on the decaying branch,
  robust to rounding (see `snell_cos_theta`).
- Frequency-independent wording: nothing is THz-specific; examples span the
  visible to the THz.

## Coding conventions

- Functions first: physics is written as plain functions (inputs in, outputs
  out). Classes are used ONLY as frozen dataclasses for returned data
  (`Result`): no methods, no inheritance, no objects that change themselves,
  no classes that only group functions (a module does that).
- Every logical block inside a function gets a one-line `#` comment saying why
  it exists. Physics code explains *why* in its docstrings (where a formula
  comes from, which branch or sign and why), with references.
- SI units need no suffix (`omega` in rad/s, thicknesses in m, angles in rad);
  non-SI or scaled values carry one (`_um`, `_nm`, `_thz`, `_deg`).
- Naming: lowercase `r`, `t` for complex field ratios, uppercase `R`, `T` for
  power fractions. Physicist-first names (`layer_n`, `theta_inc`, `eta`).
- Shared code goes in one module and is imported, never copied.
- Remove unused code instead of leaving it commented out.
- Dependencies: numpy only.
- The package (`src/cjctmm/`) and `tests/` never import cjcplot; only
  `examples/` may, via the `examples` extra pinned to a cjcplot release tag.
  When upgrading the pin, change the tag in pyproject.toml and regenerate all
  example PNGs. Examples use cjcplot's
  public API only and save their PNG with `save_path=..., fig_show=False`.
- Python 3.10 compatible (no syntax newer than 3.10).

## Dependencies

- All dependencies live in `pyproject.toml`, the only list: `[project]
  dependencies` (numpy), the `test` extra (pytest) and the `examples` extra
  (cjcplot, pinned). There is no `requirements.txt`; set up an environment
  with `pip install -e ".[test,examples]"`.

## Tests

- Tests check physical values with simple numbers that can be verified by
  hand (`r_s = -0.2` air to glass, Brewster 56.31 deg, quarter-wave `R = 0`),
  convention-free identities (`R + T = 1` without loss) and closed forms
  (Airy, Bragg, Tinkham), not just that code runs.
- Wrong shapes and invalid values raise `ValueError`s with clear messages, and
  those messages are tested.
- New or changed functionality comes with new or updated tests in the same
  change, and `pytest` passes before committing. When the public API changes,
  update the README, this file, THEORY.md and the affected examples (and their
  PNGs) too.
- Every test has a one-line docstring saying what it checks; non-obvious lines
  (the reason behind a value or tolerance) get a short comment.
- GitHub Actions (`.github/workflows/tests.yml`) runs `pytest` on Python 3.10
  and 3.14 for every push and pull request.

## Changelog and releases

- Every change that users notice (new or changed functionality, fixes, docs
  people rely on) gets a line in `CHANGELOG.md` under `## [Unreleased]`, in the
  same change, sorted under `### Added`, `### Changed`, `### Fixed` or `### Removed`.
- Before finishing a change, check that `CHANGELOG.md` describes it.
- Versions follow MAJOR.MINOR.PATCH: patch for fixes, minor for new features,
  major for changes that break existing scripts (below 1.0 the API may still change).
  A change of a convention or of returned values counts as breaking.
- A release:
  1. rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` and start a new empty
     `## [Unreleased]` above it;
  2. set the new version in `src/cjctmm/__init__.py` (`__version__`), its only
     source: `pyproject.toml` reads it from there; also update the pinned
     version in the README's install command;
  3. commit, tag `vX.Y.Z`, push the commit and the tag, and publish the GitHub
     release with the changelog section as its notes.

## Git workflow

- Do not commit, push, tag, merge or force-push without the maintainer's approval.
- Keep each change focused: no unrelated reformatting or renaming mixed in.
- Keep generated files in the same change as the code that produces them, e.g.
  regenerate an example's PNG when the example or the solver changes.
- In the commit message (or pull request), summarise what changes for users and
  how it was checked (tests run, examples regenerated).
