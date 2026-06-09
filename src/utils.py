"""Grid helpers: metre <-> cell conversion, field allocation, and the reflection map.

Cells are addressed as (row, col) so that field[row, col] indexes an (NY, NX) array
directly: row <-> y, col <-> x.
"""

import numpy as np

from grid import RoomGrid


def coord_to_cell(x: float, y: float, grid: RoomGrid) -> tuple[int, int]:
    """Metre position (x, y) -> nearest (row, col), clamped to the grid."""
    col = int(np.clip(round(x / grid.dx), 0, grid.NX - 1))
    row = int(np.clip(round(y / grid.dy), 0, grid.NY - 1))
    return row, col


def cell_to_coord(row: int, col: int, grid: RoomGrid) -> tuple[float, float]:
    """(row, col) -> metre position (x, y).  Inverse of coord_to_cell."""
    return col * grid.dx, row * grid.dy


def empty_field(grid: RoomGrid) -> np.ndarray:
    """A zero (NY, NX) float64 array — the pressure field or the alpha material map."""
    return np.zeros((grid.NY, grid.NX), dtype=np.float64)


def pressure_reflection(alpha):
    """Pressure reflection R_p = sqrt(1 - alpha) from absorption alpha.

    Absorbed-energy fraction is then 1 - R_p^2 = alpha.  alpha=0 -> R_p=1 (rigid mirror);
    alpha=1 -> R_p=0 (fully absorbing).  Works on scalars or arrays.
    """
    return np.sqrt(np.clip(1.0 - np.asarray(alpha, dtype=np.float64), 0.0, 1.0))
