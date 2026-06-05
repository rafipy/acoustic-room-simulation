"""Geometry helpers for rasterizing walls into the floor-plan grid."""

from config import FLOOR_PLAN_DX, FLOOR_PLAN_MATERIALS


def add_h_wall(grid, y_m, x0_m, x1_m, material="drywall", thickness_m=0.15):
    """Add a horizontal wall segment to the grid."""
    key = material.strip().lower()
    alpha = FLOOR_PLAN_MATERIALS.get(key)
    if alpha is None:
        raise KeyError(f"Unknown floor-plan material {material!r}; available: {sorted(FLOOR_PLAN_MATERIALS.keys())}")
    y = int(y_m / FLOOR_PLAN_DX)
    x0 = int(x0_m / FLOOR_PLAN_DX)
    x1 = int(x1_m / FLOOR_PLAN_DX)
    t = max(2, int(thickness_m / FLOOR_PLAN_DX))
    grid[y:y + t, x0:x1] = alpha


def add_v_wall(grid, x_m, y0_m, y1_m, material="drywall", thickness_m=0.15):
    """Add a vertical wall segment to the grid."""
    key = material.strip().lower()
    alpha = FLOOR_PLAN_MATERIALS.get(key)
    if alpha is None:
        raise KeyError(f"Unknown floor-plan material {material!r}; available: {sorted(FLOOR_PLAN_MATERIALS.keys())}")
    x = int(x_m / FLOOR_PLAN_DX)
    y0 = int(y0_m / FLOOR_PLAN_DX)
    y1 = int(y1_m / FLOOR_PLAN_DX)
    t = max(2, int(thickness_m / FLOOR_PLAN_DX))
    grid[y0:y1, x:x + t] = alpha


def add_rect(grid, x0_m, y0_m, x1_m, y1_m, material="drywall"):
    """Fill a rectangular region with a material."""
    key = material.strip().lower()
    alpha = FLOOR_PLAN_MATERIALS.get(key)
    if alpha is None:
        raise KeyError(f"Unknown floor-plan material {material!r}; available: {sorted(FLOOR_PLAN_MATERIALS.keys())}")
    x0, x1 = int(x0_m / FLOOR_PLAN_DX), int(x1_m / FLOOR_PLAN_DX)
    y0, y1 = int(y0_m / FLOOR_PLAN_DX), int(y1_m / FLOOR_PLAN_DX)
    grid[y0:y1, x0:x1] = alpha
