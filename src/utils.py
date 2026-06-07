
import numpy as np
from grid import RoomGrid


# ── Coordinate conversion ────────────────────────────────────────────────────
# Cells are addressed as (row, col) = (j, i) so that  field[row, col]  indexes an
# (NY, NX) array directly:  row j ↔ y,  col i ↔ x.

def coord_to_cell(x: float, y: float, grid: RoomGrid) -> tuple[int, int]:
    """
    Convert a physical position (x, y) in metres to the nearest grid cell (row, col).
    Clamps to the valid index range, so out-of-bounds positions map onto the boundary.
    """
    col = int(np.clip(round(x / grid.dx), 0, grid.NX - 1))
    row = int(np.clip(round(y / grid.dy), 0, grid.NY - 1))
    return row, col


def cell_to_coord(row: int, col: int, grid: RoomGrid) -> tuple[float, float]:
    """
    Convert grid indices (row, col) to physical coordinates (x, y) in metres.
    Inverse of coord_to_cell.
    """
    return col * grid.dx, row * grid.dy


# ── Grid-line generators ─────────────────────────────────────────────────────

def x_lines(grid: RoomGrid) -> np.ndarray:
    """X positions (metres) of all vertical grid lines — 1D array of length NX."""
    return np.arange(grid.NX) * grid.dx


def y_lines(grid: RoomGrid) -> np.ndarray:
    """Y positions (metres) of all horizontal grid lines — 1D array of length NY."""
    return np.arange(grid.NY) * grid.dy


def grid_lines(grid: RoomGrid) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_positions, y_positions) for all grid lines."""
    return x_lines(grid), y_lines(grid)


# ── Field factory ─────────────────────────────────────────────────────────────

def empty_field(grid: RoomGrid) -> np.ndarray:
    """
    Create a zero-initialised (NY, NX) float64 array.
    Used to allocate the pressure field and the material/α map.
    """
    return np.zeros((grid.NY, grid.NX), dtype=np.float64)


# ── Material → reflection ─────────────────────────────────────────────────────

def pressure_reflection(alpha):
    """
    Pressure reflection coefficient  R_p = √(1 − α)  from an absorption coefficient α.

    Works on scalars or NumPy arrays.  By construction the fraction of incident
    ENERGY absorbed at the surface is  1 − R_p² = α, so a tabulated absorption
    coefficient maps directly onto wall behaviour:
        α = 0  → R_p = 1   (perfect rigid reflector)
        α = 1  → R_p = 0   (no reflected wave → fully absorbing)
    """
    return np.sqrt(np.clip(1.0 - np.asarray(alpha, dtype=np.float64), 0.0, 1.0))


# ── Wall helpers ──────────────────────────────────────────────────────────────
# With the per-cell α map (room.py), the primary "is this a wall?" test is simply
# alpha > 0.  These outer-frame helpers are kept for convenience and the grid
# sanity-check notebook.

WALL_SIDES = ("N", "S", "E", "W")


def boundary_mask(grid: RoomGrid) -> np.ndarray:
    """
    Return an (NY, NX) bool array, True on every outer-frame cell.
    row 0 = South (y=0), row NY-1 = North; col 0 = West (x=0), col NX-1 = East.
    """
    mask = np.zeros((grid.NY, grid.NX), dtype=bool)
    mask[ 0, :] = True   # South (j = 0, y = 0)
    mask[-1, :] = True   # North (j = NY-1)
    mask[:,  0] = True   # West  (i = 0, x = 0)
    mask[:, -1] = True   # East  (i = NX-1)
    return mask


def interior_mask(grid: RoomGrid) -> np.ndarray:
    """Return an (NY, NX) bool array, True on every non-boundary cell."""
    return ~boundary_mask(grid)


def wall_cells(grid: RoomGrid, side: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (rows, cols) index arrays for one outer-frame side.
    side: one of "N", "S", "E", "W".  Compatible with fancy indexing:
        field[wall_cells(grid, "N")]
    """
    NX, NY = grid.NX, grid.NY
    if side == "N":
        return np.full(NX, NY - 1), np.arange(NX)
    if side == "S":
        return np.zeros(NX, dtype=int), np.arange(NX)
    if side == "E":
        return np.arange(NY), np.full(NY, NX - 1)
    if side == "W":
        return np.arange(NY), np.zeros(NY, dtype=int)
    raise ValueError(f"side must be one of {WALL_SIDES}, got {side!r}")


# ── Quick check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import MATERIALS

    grid = RoomGrid()
    print(grid)
    print()

    # Coordinate round-trip
    x, y     = 3.5, 2.0
    row, col = coord_to_cell(x, y, grid)
    xb, yb   = cell_to_coord(row, col, grid)
    print(f"({x}, {y}) m  ->  cell (row={row}, col={col})  ->  ({xb:.4f}, {yb:.4f}) m")
    print()

    xs, ys = grid_lines(grid)
    print(f"Vertical lines   : {len(xs)} lines  x in [{xs[0]:.3f}, {xs[-1]:.3f}] m")
    print(f"Horizontal lines : {len(ys)} lines  y in [{ys[0]:.3f}, {ys[-1]:.3f}] m")
    print()

    field = empty_field(grid)
    print(f"Empty field shape : {field.shape}  dtype: {field.dtype}   (NY, NX) = ({grid.NY}, {grid.NX})")
    print()

    print("Material -> reflection (R_p = sqrt(1-alpha)):")
    for name, a in MATERIALS.items():
        Rp = float(pressure_reflection(a))
        print(f"  {name:>14}  alpha={a:.2f}  ->  R_p={Rp:.3f}   (absorbs {a*100:.0f}% of incident energy)")
    print()

    bm = boundary_mask(grid)
    n_boundary = int(bm.sum())
    n_interior = bm.size - n_boundary
    print(f"Boundary cells : {n_boundary}")
    print(f"Interior cells : {n_interior}  (sum = {n_boundary + n_interior} = NX*NY = {grid.NX * grid.NY})")
    print()

    for side in WALL_SIDES:
        rows, cols = wall_cells(grid, side)
        print(f"Wall {side}: {len(rows)} cells, head = (rows[:3]={rows[:3].tolist()}, cols[:3]={cols[:3].tolist()})")
