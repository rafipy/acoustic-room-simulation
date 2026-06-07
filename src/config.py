
# Physical constants
C_SOUND: float = 343.0   # m/s  (speed of sound in air at ~20 °C)

# Source / resolution frequency.
# The "clap" is a broadband Gaussian pulse, not a pure tone — but we still need a
# characteristic frequency to (a) size the grid (PPW points across one wavelength)
# and (b) shape the source pulse so its spectrum stays resolved by the grid.
# Read FREQ as "the highest frequency we want to resolve cleanly".
FREQ:    float = 440.0   # Hz

# Room dimensions
ROOM_W: float = 8.0      # metres
ROOM_H: float = 6.0      # metres

# Grid resolution
PPW: int = 12            # points per wavelength (≥ ~10 keeps numerical dispersion low)

# Time step.  CFL_FACTOR is used directly as the Courant number r = c·dt/dx in RoomGrid.
# The 2D stability limit is r ≤ 1/√2 ≈ 0.707; we run at 0.9 of that limit (0.9/√2 ≈ 0.636).
# Smaller values (e.g. 0.40) are more conservative: less numerical dispersion, more steps.
CFL_FACTOR: float = 0.636

# Air (volumetric) damping coefficient β in   u_tt + β·u_t = c²∇²u.
# Default 0 → lossless air; ALL energy loss happens at walls/furniture (their α).
# The solver keeps β as a tunable knob — raise it slightly for gentle global decay.
BETA: float = 0.0

# Material presets — α = ENERGY ABSORPTION coefficient (0 = perfect reflector, 1 = perfect absorber).
# Values are mid-frequency (~500 Hz) Sabine absorption coefficients from standard acoustic
# tables (ISO 354).  The solver converts each α into a pressure reflection coefficient
#     R_p = √(1 − α)
# so that the fraction of incident ENERGY absorbed at a surface is exactly  1 − R_p² = α.
# (see utils.pressure_reflection and the physics_solver derivation.)
MATERIALS: dict[str, float] = {
    "Concrete":      0.02,
    "Brick":         0.03,
    "Plaster":       0.03,
    "Glass":         0.05,
    "Wood":          0.15,
    "Carpet":        0.40,
    "Heavy Curtain": 0.55,
    "Acoustic Foam": 0.85,
    "Open":          1.00,   # anechoic / fully absorbing boundary (R_p = 0)
}
