"""
main.py — end-to-end smoke test of the physics pipeline.

Builds a room, drops a "clap", steps the FDTD solver, and prints field stats.
The interactive Pygame visualiser (visualiser.py) is owned by the other teammate
and simply renders the pressure field this pipeline produces.
"""

import sys

import numpy as np

from grid import RoomGrid
from room import Room
from physics_solver import WaveSolver

# TODO (visualiser): from visualiser import render   # owned by teammate


def build_room(grid: RoomGrid) -> Room:
    """A two-room layout: concrete shell, wooden partition + doorway, a sofa."""
    return (Room(grid)
            .add_border("Concrete")
            .add_rectangle("Wood", 4.0, 0.0, 4.15, 6.0)   # partition wall at x ~ 4 m
            .add_doorway(4.0, 2.5, 4.15, 3.5)             # 1 m doorway gap
            .add_block("Carpet", 1.0, 1.0, 2.5, 2.0))     # soft furniture (sofa)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    grid = RoomGrid()
    print(grid)

    room = build_room(grid)
    print(room.summary())

    solver = WaveSolver(grid, room.alpha)
    solver.add_impulse(2.0, 3.0)        # clap in the left room
    solver.run(600)

    f = solver.field
    print(f"\nafter {solver.n} steps  (t = {solver.t * 1e3:.2f} ms):")
    print(f"  finite        : {bool(np.all(np.isfinite(f)))}")
    print(f"  max |u|       : {float(np.max(np.abs(f))):.4f}")
    print(f"  energy (Su^2) : {solver.energy():.4f}")

    # TODO (visualiser): render(solver, ...) — animate the pressure field over time


if __name__ == "__main__":
    main()
