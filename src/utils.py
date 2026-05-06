
import numpy as np
from grid import RoomGrid


# ── Coordinate conversion ────────────────────────────────────────────────────

def coord_to_cell(x: float, y: float, grid: RoomGrid) -> tuple[int, int]:
    """
    Convert a physical position (x, y) in metres to the nearest grid cell (i, j).
    Clamps to valid index range so out-of-bounds positions map to the boundary.
    """
    i = int(np.clip(round(x / grid.dx), 0, grid.NX - 1))
    j = int(np.clip(round(y / grid.dy), 0, grid.NY - 1))
    return i, j


def cell_to_coord(i: int, j: int, grid: RoomGrid) -> tuple[float, float]:
    """
    Convert grid indices (i, j) to physical coordinates (x, y) in metres.
    Returns the centre of the cell.
    """
    return i * grid.dx, j * grid.dy


# ── Grid-line generators ─────────────────────────────────────────────────────

def x_lines(grid: RoomGrid) -> np.ndarray:
    """
    X positions (metres) of all vertical grid lines.
    Returns a 1D array of length NX.
    """
    return np.arange(grid.NX) * grid.dx


def y_lines(grid: RoomGrid) -> np.ndarray:
    """
    Y positions (metres) of all horizontal grid lines.
    Returns a 1D array of length NY.
    """
    return np.arange(grid.NY) * grid.dy


def grid_lines(grid: RoomGrid) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (x_positions, y_positions) for all grid lines.
    Convenience wrapper around x_lines / y_lines.
    """
    return x_lines(grid), y_lines(grid)


# ── Field factory ─────────────────────────────────────────────────────────────

def empty_field(grid: RoomGrid) -> np.ndarray:
    """
    Create a zero-initialised (NX × NY) float64 array.
    Used to allocate pressure or any other scalar field on the grid.
    """
    return np.zeros((grid.NX, grid.NY), dtype=np.float64)


# ── Wall helpers ──────────────────────────────────────────────────────────────

WALL_SIDES = ("N", "S", "E", "W")


def boundary_mask(grid: RoomGrid) -> np.ndarray:
    """
    Return an (NX × NY) bool array, True on every wall cell.
    The FDTD update will skip these cells; absorption is applied via grid.alpha.
    """
    mask = np.zeros((grid.NX, grid.NY), dtype=bool)
    mask[ 0, :] = True   # West  (i = 0)
    mask[-1, :] = True   # East  (i = NX - 1)
    mask[:,  0] = True   # South (j = 0)
    mask[:, -1] = True   # North (j = NY - 1)
    return mask


def interior_mask(grid: RoomGrid) -> np.ndarray:
    """
    Return an (NX × NY) bool array, True on every non-boundary cell.
    Convenience for the FDTD update which only writes interior cells.
    """
    return ~boundary_mask(grid)


def wall_cells(grid: RoomGrid, side: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (i_indices, j_indices) for the requested wall side.
    side: one of "N", "S", "E", "W".
    Output shape is compatible with NumPy fancy indexing: field[wall_cells(g, "N")].
    """
    NX, NY = grid.NX, grid.NY
    if side == "N":
        return np.arange(NX), np.full(NX, NY - 1)
    if side == "S":
        return np.arange(NX), np.zeros(NX, dtype=int)
    if side == "E":
        return np.full(NY, NX - 1), np.arange(NY)
    if side == "W":
        return np.zeros(NY, dtype=int), np.arange(NY)
    raise ValueError(f"side must be one of {WALL_SIDES}, got {side!r}")


# ── Quick check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import MATERIALS

    grid = RoomGrid()
    print(grid)
    print()

    # Coordinate round-trip
    x, y   = 3.5, 2.0
    i, j   = coord_to_cell(x, y, grid)
    xb, yb = cell_to_coord(i, j, grid)
    print(f"({x}, {y}) m  →  cell ({i}, {j})  →  ({xb:.4f}, {yb:.4f}) m")
    print()

    xs, ys = grid_lines(grid)
    print(f"Vertical lines   : {len(xs)} lines  x ∈ [{xs[0]:.3f}, {xs[-1]:.3f}] m")
    print(f"Horizontal lines : {len(ys)} lines  y ∈ [{ys[0]:.3f}, {ys[-1]:.3f}] m")
    print()

    field = empty_field(grid)
    print(f"Empty field shape : {field.shape}  dtype: {field.dtype}")
    print()

    bm = boundary_mask(grid)
    n_boundary = int(bm.sum())
    n_interior = bm.size - n_boundary
    print(f"Boundary cells : {n_boundary}")
    print(f"Interior cells : {n_interior}  (boundary + interior = {n_boundary + n_interior} = NX*NY = {grid.NX * grid.NY})")
    print()

    for side in WALL_SIDES:
        i, j = wall_cells(grid, side)
        print(f"Wall {side}: {len(i)} cells, head = (i[:3]={i[:3].tolist()}, j[:3]={j[:3].tolist()})")
