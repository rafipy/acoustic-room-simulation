from config import MATERIALS
from grid import RoomGrid
from utils import WALL_SIDES, boundary_mask, wall_cells

# TODO (physics):    from physics_solver import step           # owned by Rafie
# TODO (visualiser): from visualiser     import render         # owned by teammate


def main():
    grid = RoomGrid(alpha=MATERIALS["Wood"])
    print(grid)
    print()

    bm = boundary_mask(grid)
    n_boundary = int(bm.sum())
    n_interior = bm.size - n_boundary
    print(f"Boundary cells : {n_boundary}")
    print(f"Interior cells : {n_interior}")
    assert n_boundary + n_interior == grid.NX * grid.NY
    print()

    for side in WALL_SIDES:
        i, j = wall_cells(grid, side)
        print(f"Wall {side}: {len(i)} cells")

    # TODO (physics):    step(grid, source, ...)   — drive the wave equation
    # TODO (visualiser): render(grid, frames, ...) — animate the pressure field


if __name__ == "__main__":
    main()
