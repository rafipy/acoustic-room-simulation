
# Physical constants
C_SOUND: float = 343.0   # m/s  (speed of sound in air at ~20 °C)
FREQ:    float = 440.0   # Hz   (source frequency; also drives grid resolution)

# Room dimensions
# Set to 6×6 to match the floor-plan canvas request
ROOM_W: float = 6.0      # metres
ROOM_H: float = 6.0      # metres

# Grid resolution
PPW: int = 12             # points per wavelength
CFL_FACTOR: float = 0.40  # safety factor for time step (limit is 1/√2 ≈ 0.707)

# Floor plan visualisation — derive room size from existing ROOM_W / ROOM_H
# so the floor-plan "canvas" uses the project's canonical room dimensions.
FLOOR_PLAN_ROOM_WIDTH_M: float = ROOM_W
FLOOR_PLAN_ROOM_HEIGHT_M: float = ROOM_H
FLOOR_PLAN_DX: float = 0.05

# Canvas grid resolution (number of cells horizontally / vertically)
FLOOR_PLAN_NX: int = int(FLOOR_PLAN_ROOM_WIDTH_M / FLOOR_PLAN_DX)
FLOOR_PLAN_NY: int = int(FLOOR_PLAN_ROOM_HEIGHT_M / FLOOR_PLAN_DX)

FLOOR_PLAN_CELL_SIZE: int = 6

FLOOR_PLAN_MARGIN_LEFT: int = 55
FLOOR_PLAN_MARGIN_RIGHT: int = 115
FLOOR_PLAN_MARGIN_TOP: int = 50
FLOOR_PLAN_MARGIN_BOTTOM: int = 50

FLOOR_PLAN_GRID_PX_W: int = FLOOR_PLAN_NX * FLOOR_PLAN_CELL_SIZE
FLOOR_PLAN_GRID_PX_H: int = FLOOR_PLAN_NY * FLOOR_PLAN_CELL_SIZE
FLOOR_PLAN_SCREEN_W: int = (
    FLOOR_PLAN_MARGIN_LEFT + FLOOR_PLAN_GRID_PX_W + FLOOR_PLAN_MARGIN_RIGHT
)
FLOOR_PLAN_SCREEN_H: int = (
    FLOOR_PLAN_MARGIN_TOP + FLOOR_PLAN_GRID_PX_H + FLOOR_PLAN_MARGIN_BOTTOM
)

# Floor-plan-specific material lookup. Populated from the canonical
# `MATERIALS` mapping (lowercased keys) further below to avoid duplication.
FLOOR_PLAN_MATERIALS: dict[str, float] = {}

# Material presets
# α = absorption coefficient (0 = perfect reflector, 1 = perfect absorber)
# Values from standard acoustic tables at ~500 Hz (ISO 354)
MATERIALS: dict[str, float] = {
    "Concrete":      0.02,
    "Brick":         0.03,
    "Glass":         0.05,
    "Drywall":       0.03,
    "Wood":          0.15,
    "Carpet":        0.40,
    "Acoustic Foam": 0.85,
}

# Create a lowercase-keyed view for the floor-plan code which expects
# material names like "concrete", "glass", "drywall". This keeps the
# canonical `MATERIALS` mapping intact for other modules while avoiding
# duplicate hard-coded values.
FLOOR_PLAN_MATERIALS.update({k.lower(): v for k, v in MATERIALS.items()})
