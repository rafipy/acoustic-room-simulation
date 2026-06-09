"""Physical constants and material presets — the single source of truth."""

# Acoustics
C_SOUND: float = 343.0    # speed of sound in air (m/s, ~20 C)
FREQ:    float = 440.0    # resolution frequency: sizes the grid and shapes the source pulse

# Room (metres) and grid resolution
ROOM_W: float = 8.0
ROOM_H: float = 6.0
PPW:    int   = 12         # points per wavelength (>= ~10 limits numerical dispersion)

# Courant number r = c*dt/dx, used directly by RoomGrid. 2D stability needs r <= 1/sqrt(2).
CFL_FACTOR: float = 0.636  # 0.9 of the limit

# Air damping beta in  u_tt + beta*u_t = c^2 * lap(u).  0 = lossless air (all loss at walls).
BETA: float = 0.0

# alpha = energy absorption coefficient (0 = perfect reflector, 1 = perfect absorber),
# mid-frequency Sabine values (ISO 354).  The solver turns alpha into a pressure reflection
# coefficient R_p = sqrt(1 - alpha), so the absorbed-energy fraction is exactly 1 - R_p^2 = alpha.
MATERIALS: dict[str, float] = {
    "Concrete":      0.02,
    "Brick":         0.03,
    "Plaster":       0.03,
    "Drywall":       0.05,
    "Glass":         0.05,
    "Wood":          0.15,
    "Carpet":        0.40,
    "Heavy Curtain": 0.55,
    "Acoustic Foam": 0.85,
    "Open":          1.00,   # anechoic / fully absorbing (R_p = 0)
}
