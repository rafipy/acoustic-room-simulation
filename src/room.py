"""
room.py — imperative scene builder for the acoustic room.

Produces the (NY, NX) material map (the "alpha-map") that the FDTD solver consumes:
    alpha[row, col] == 0.0   ->  air  (the wave propagates freely)
    alpha[row, col]  > 0.0   ->  obstacle (wall / furniture) with that ABSORPTION
                                 coefficient α; the solver reflects with R_p = √(1-α).

All geometry is given in METRES and converted to cell indices via utils.coord_to_cell
(which clamps to the grid).  Builders return self, so calls chain:

    room = (Room(grid)
            .add_border("Concrete")                       # closed outer shell
            .add_rectangle("Wood", 4.0, 0.0, 4.15, 6.0)   # interior partition
            .add_doorway(4.0, 2.5, 4.15, 3.5)             # gap in that partition
            .add_block("Carpet", 1.0, 1.0, 2.5, 2.0))     # absorbing furniture

This module is a sibling of physics_solver.py: both build only on config/grid/utils
and never import each other.  The solver receives `room.alpha`, not the Room object.

NOTE: a perfectly rigid obstacle is α≈0 (e.g. Concrete 0.02 → R_p≈0.99).  Exactly
α=0 means "air", so it is NOT an obstacle — use a small α for a hard wall.
"""

import numpy as np

from grid import RoomGrid
from utils import empty_field, coord_to_cell
from config import MATERIALS


def _alpha_value(material) -> float:
    """Resolve a material spec to an absorption coefficient α.

    Accepts a preset name ("Wood") or a raw float α in [0, 1].
    """
    if isinstance(material, str):
        return MATERIALS[material]
    a = float(material)
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {a}")
    return a


class Room:
    """A mutable (NY, NX) absorption map you stamp walls and furniture into."""

    def __init__(self, grid: RoomGrid):
        self.grid = grid
        self.alpha = empty_field(grid)          # all air (0.0) to start

    # ── helpers ──────────────────────────────────────────────────────────────
    def _cell_box(self, x0, y0, x1, y1):
        """Metre rectangle -> inclusive (row0, row1, col0, col1) cell-index box."""
        r0, c0 = coord_to_cell(min(x0, x1), min(y0, y1), self.grid)
        r1, c1 = coord_to_cell(max(x0, x1), max(y0, y1), self.grid)
        return r0, r1, c0, c1

    # ── builders ─────────────────────────────────────────────────────────────
    def add_rectangle(self, material, x0, y0, x1, y1):
        """Fill a metre rectangle with an obstacle material (wall or furniture)."""
        a = _alpha_value(material)
        r0, r1, c0, c1 = self._cell_box(x0, y0, x1, y1)
        self.alpha[r0:r1 + 1, c0:c1 + 1] = a
        return self

    # furniture reads more naturally as "a block"; same operation
    add_block = add_rectangle

    def add_border(self, material, thickness: int = 1):
        """Stamp the outer frame (closed room).  `thickness` is in cells (>= 1)."""
        a = _alpha_value(material)
        t = max(1, int(thickness))
        self.alpha[:t, :]  = a   # South edge (y = 0)
        self.alpha[-t:, :] = a   # North edge
        self.alpha[:, :t]  = a   # West edge  (x = 0)
        self.alpha[:, -t:] = a   # East edge
        return self

    def carve(self, x0, y0, x1, y1):
        """Reset a metre rectangle back to air (α = 0) — e.g. a doorway gap."""
        r0, r1, c0, c1 = self._cell_box(x0, y0, x1, y1)
        self.alpha[r0:r1 + 1, c0:c1 + 1] = 0.0
        return self

    # a doorway is simply a carved gap in a wall
    add_doorway = carve

    # ── views ────────────────────────────────────────────────────────────────
    @property
    def is_solid(self) -> np.ndarray:
        """Boolean (NY, NX) mask: True wherever there is an obstacle."""
        return self.alpha > 0.0

    def summary(self) -> str:
        solid = int(self.is_solid.sum())
        present = sorted({round(float(v), 3) for v in np.unique(self.alpha) if v > 0})
        return (f"Room {self.grid.NX} x {self.grid.NY} cells | "
                f"{solid} solid ({100 * solid / self.alpha.size:.1f}%) | "
                f"alpha present: {present}")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    grid = RoomGrid()

    # Two-room layout: concrete shell, a wooden partition at x≈4 m with a 1 m
    # doorway, and an absorbing furniture block (a sofa) in the left room.
    room = (Room(grid)
            .add_border("Concrete")
            .add_rectangle("Wood", 4.0, 0.0, 4.15, 6.0)   # partition wall
            .add_doorway(4.0, 2.5, 4.15, 3.5)             # doorway gap
            .add_block("Carpet", 1.0, 1.0, 2.5, 2.0))     # soft furniture (sofa)

    print(room.summary())
    print()

    # Spot-checks (verification: builders stamp the expected α) -----------------
    checks = [
        ("partition (x=4.07,y=5.0)", (4.07, 5.0), 0.15, "Wood"),
        ("doorway   (x=4.07,y=3.0)", (4.07, 3.0), 0.00, "air"),
        ("furniture (x=1.5, y=1.5)", (1.5, 1.5),  0.40, "Carpet"),
        ("open air  (x=6.0, y=5.0)", (6.0, 5.0),  0.00, "air"),
    ]
    ok = True
    for label, (x, y), expect, name in checks:
        r, c = coord_to_cell(x, y, grid)
        got = float(room.alpha[r, c])
        match = abs(got - expect) < 1e-9
        ok = ok and match
        print(f"  {label}: alpha={got:.2f}  expect {expect:.2f} ({name})  "
              f"{'OK' if match else 'MISMATCH'}")
    print(f"  border    (row 0)        : alpha={float(room.alpha[0, 60]):.2f}  "
          f"expect 0.02 (Concrete)")
    print("\nPASS" if ok else "\nFAIL")
