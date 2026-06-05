import numpy as np
from config import FLOOR_PLAN_NX, FLOOR_PLAN_NY, FLOOR_PLAN_ROOM_HEIGHT_M, FLOOR_PLAN_ROOM_WIDTH_M
from floor_plan_geometry import add_h_wall, add_v_wall
from floor_plan_renderer import show_grid


# ── Floor plan construction ───────────────────────────────────────────────────

def create_floor_plan() -> np.ndarray:
    """
    Build the absorption coefficient grid for the tiny home floor plan.

    Returns:
        (NY, NX) float array where 0.0 = open air, non-zero = wall/material cell.
    """
    grid    = np.zeros((FLOOR_PLAN_NY, FLOOR_PLAN_NX), dtype=float)
    W       = FLOOR_PLAN_ROOM_WIDTH_M
    H       = FLOOR_PLAN_ROOM_HEIGHT_M
    wall_th = 0.1

    # ── Outer walls ───────────────────────────────────────────────────────────
    # Top wall: concrete with one glass window segment near the left
    add_h_wall(grid, H - wall_th, wall_th, 0.75,        material="concrete")
    add_h_wall(grid, H - wall_th, 0.75,   1.35,         material="glass")
    add_h_wall(grid, H - wall_th, 1.35,   W - wall_th,  material="concrete")

    # Left wall: fully solid concrete
    add_v_wall(grid, 0.0, 0.0, H, material="concrete")

    # Right wall: two vertical window openings separated by concrete piers
    add_v_wall(grid, W - wall_th, 0.00, 0.40, material="concrete")
    add_v_wall(grid, W - wall_th, 0.40, 1.00, material="glass")
    add_v_wall(grid, W - wall_th, 1.00, 2.15, material="concrete")
    add_v_wall(grid, W - wall_th, 2.15, 2.75, material="glass")
    add_v_wall(grid, W - wall_th, 2.75, 4.00, material="concrete")
    add_v_wall(grid, W - wall_th, 4.00, 5.00, material="glass")
    add_v_wall(grid, W - wall_th, 5.00, H,    material="concrete")

    # Bottom wall: window near middle-left, front door opening on the right
    add_h_wall(grid, 0.0, wall_th, 0.80, material="concrete")
    add_h_wall(grid, 0.0, 0.80,   1.40,  material="glass")
    add_h_wall(grid, 0.0, 1.40,   3.00,  material="concrete")
    add_h_wall(grid, 0.0, 3.00,   3.85,  material="glass")
    add_h_wall(grid, 0.0, 3.85,   4.25,  material="concrete")
    add_h_wall(grid, 0.0, 5.25,   H - wall_th, material="concrete")
    # intentional gap 4.25 → 5.25: front door opening

    # ── Inner walls ───────────────────────────────────────────────────────────
    # Main horizontal divider between upper rooms and the living/kitchen area
    add_h_wall(grid, 3.0, wall_th, 3.20,        material="drywall")
    add_h_wall(grid, 3.0, 5.00,   W - wall_th,  material="drywall")

    # Upper-left bath / utility / pantry partitions
    add_v_wall(grid, 1.75, 3.15, 4.75, material="drywall")
    add_h_wall(grid, 4.80, 1.75, 2.25, material="drywall")
    add_v_wall(grid, 0.65, 2.45, 3.00, material="drywall")
    add_v_wall(grid, 1.75, 2.45, 3.00, material="drywall")

    # Small detail walls (short stubs visible in reference image)
    add_v_wall(grid, 2.20, 3.05 + wall_th, 3.25, material="drywall")
    add_v_wall(grid, 2.20, 4.65, 5.00,           material="drywall")
    add_v_wall(grid, 2.20, 5.80, H - wall_th,    material="drywall")
    # add_h_wall(grid, 4.35, 1.65, 2.85, material="drywall")  # kept commented out

    return grid


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    wall_grid = create_floor_plan()
    show_grid(wall_grid, title="Floor Plan - Acoustic Absorption")