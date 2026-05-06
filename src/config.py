
# Physical constants
C_SOUND: float = 343.0   # m/s  (speed of sound in air at ~20 °C)
FREQ:    float = 440.0   # Hz   (source frequency; also drives grid resolution)

# Room dimensions
ROOM_W: float = 8.0      # metres
ROOM_H: float = 6.0      # metres

# Grid resolution
PPW: int = 12             # points per wavelength
CFL_FACTOR: float = 0.40  # safety factor for time step (limit is 1/√2 ≈ 0.707)

# Material presets
# α = absorption coefficient (0 = perfect reflector, 1 = perfect absorber)
# Values from standard acoustic tables at ~500 Hz (ISO 354)
MATERIALS: dict[str, float] = {
    "Concrete":      0.02,
    "Brick":         0.03,
    "Glass":         0.05,
    "Wood":          0.15,
    "Carpet":        0.40,
    "Acoustic Foam": 0.85,
}
