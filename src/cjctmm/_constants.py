"""Physical constants in SI units.

Written out here instead of importing scipy, so numpy stays the only dependency.
"""

# Speed of light in vacuum [m/s], exact by the SI definition of the metre.
SPEED_OF_LIGHT = 299_792_458.0

# Vacuum magnetic permeability mu0 [N/A^2], CODATA 2022. Since the 2019 SI
# redefinition it is measured, not exactly 4*pi*1e-7 (they differ by ~1e-10).
VACUUM_PERMEABILITY = 1.25663706127e-6

# Vacuum electric permittivity eps0 = 1/(mu0*c^2) [F/m].
VACUUM_PERMITTIVITY = 1.0 / (VACUUM_PERMEABILITY * SPEED_OF_LIGHT**2)

# Impedance of free space Z0 = mu0*c ~ 376.73 ohm: the ratio |E|/|H| of a plane
# wave in vacuum, and the unit in which admittances n/Z0 are expressed.
VACUUM_IMPEDANCE = VACUUM_PERMEABILITY * SPEED_OF_LIGHT
