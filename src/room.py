"""
room.py — builds the (NY, NX) alpha material map the solver consumes (0 = air, >0 = obstacle).

Geometry is given in metres.  Builders chain, and each registers a selectable *piece*
(stamped into a parallel piece_id map) so a frontend can recolour (set_material /
set_wall_material), build (add_rectangle / add_block), carve (add_doorway), or delete
(remove_piece) pieces; the outer shell is protected.  After any edit call
solver.set_alpha(room.alpha).  Use a small alpha for a hard wall — exactly 0 means air.
"""

import numpy as np

from grid import RoomGrid
from utils import empty_field, coord_to_cell
from config import MATERIALS


def _alpha_value(material) -> float:
    """Resolve a material spec (preset name or raw float alpha in [0, 1]) to its alpha."""
    if isinstance(material, str):
        return MATERIALS[material]
    a = float(material)
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {a}")
    return a


class Piece:
    """A named, recolourable obstacle (a wall or a piece of furniture)."""
    __slots__ = ("name", "material", "kind", "protected")

    def __init__(self, name, material, kind="wall", protected=False):
        self.name = name
        self.material = material      # preset name (str) or raw float alpha
        self.kind = kind              # "wall" | "furniture"
        self.protected = protected    # protected pieces (the outer shell) can't be deleted

    @property
    def alpha(self) -> float:
        return _alpha_value(self.material)

    def label(self) -> str:
        m = self.material if isinstance(self.material, str) else f"a={self.material:.2f}"
        return f"{self.name} ({m})"


class Room:
    """A mutable (NY, NX) absorption map you stamp walls and furniture into."""

    def __init__(self, grid: RoomGrid):
        self.grid = grid
        self.alpha = empty_field(grid)                                   # 0.0 = air
        self.piece_id = np.full((grid.NY, grid.NX), -1, dtype=np.int32)  # -1 = air
        self.pieces: list[Piece] = []                                    # index == piece id
        self.locked = np.zeros((grid.NY, grid.NX), dtype=bool)           # protected cells

    # ── helpers ──────────────────────────────────────────────────────────────
    def _cell_box(self, x0, y0, x1, y1):
        r0, c0 = coord_to_cell(min(x0, x1), min(y0, y1), self.grid)
        r1, c1 = coord_to_cell(max(x0, x1), max(y0, y1), self.grid)
        return r0, r1, c0, c1

    def _rect_mask(self, x0, y0, x1, y1) -> np.ndarray:
        r0, r1, c0, c1 = self._cell_box(x0, y0, x1, y1)
        m = np.zeros_like(self.piece_id, dtype=bool)
        m[r0:r1 + 1, c0:c1 + 1] = True
        return m

    def _stamp(self, name, material, mask, kind="wall", protected=False) -> int:
        """Register a piece over mask.  Protected (shell) cells are never overwritten,
        so building near the edge can't 'steal' the outer walls."""
        if not protected:
            mask = mask & ~self.locked
        idx = len(self.pieces)
        self.pieces.append(Piece(name, material, kind, protected))
        self.alpha[mask] = _alpha_value(material)
        self.piece_id[mask] = idx
        if protected:
            self.locked |= mask
        return idx

    # ── builders ─────────────────────────────────────────────────────────────
    def add_rectangle(self, material, x0, y0, x1, y1, name=None, kind="wall", protected=False):
        """Fill a metre rectangle with an obstacle material (a wall by default)."""
        self._stamp(name or f"{kind} {len(self.pieces)}", material,
                    self._rect_mask(x0, y0, x1, y1), kind=kind, protected=protected)
        return self

    def add_block(self, material, x0, y0, x1, y1, name=None):
        """Fill a metre rectangle with FURNITURE (recoloured individually)."""
        return self.add_rectangle(material, x0, y0, x1, y1, name=name, kind="furniture")

    def add_border(self, material, thickness: int = 1, name="border"):
        """Stamp the protected outer frame (closed room).  thickness is in cells (>= 1)."""
        t = max(1, int(thickness))
        m = np.zeros_like(self.piece_id, dtype=bool)
        m[:t, :] = True
        m[-t:, :] = True
        m[:, :t] = True
        m[:, -t:] = True
        self._stamp(name, material, m, kind="wall", protected=True)
        return self

    def carve(self, x0, y0, x1, y1):
        """Reset a metre rectangle back to air — e.g. a doorway gap.  Shell cells stay intact."""
        m = self._rect_mask(x0, y0, x1, y1) & ~self.locked
        self.alpha[m] = 0.0
        self.piece_id[m] = -1
        return self

    add_doorway = carve

    # ── editing ────────────────────────────────────────────────────────────────
    def piece_at(self, row, col):
        """Piece id at a cell, or None if it is air / out of bounds."""
        if not (0 <= row < self.grid.NY and 0 <= col < self.grid.NX):
            return None
        idx = int(self.piece_id[row, col])
        return idx if idx >= 0 else None

    def set_material(self, idx, material):
        """Recolour a single piece."""
        if 0 <= idx < len(self.pieces):
            self.pieces[idx].material = material
            self.alpha[self.piece_id == idx] = _alpha_value(material)
        return self

    def set_wall_material(self, material):
        """Recolour every wall-kind piece at once (walls behave as one group)."""
        a = _alpha_value(material)
        for idx, p in enumerate(self.pieces):
            if p.kind == "wall":
                p.material = material
                self.alpha[self.piece_id == idx] = a
        return self

    def remove_piece(self, idx):
        """Delete a piece (cells return to air).  Protected pieces are kept; ids stay stable."""
        if 0 <= idx < len(self.pieces) and not self.pieces[idx].protected:
            mask = self.piece_id == idx
            self.alpha[mask] = 0.0
            self.piece_id[mask] = -1
        return self

    # ── undo support ─────────────────────────────────────────────────────────
    def snapshot(self):
        return (self.alpha.copy(), self.piece_id.copy(), self.locked.copy(),
                [(p.name, p.material, p.kind, p.protected) for p in self.pieces])

    def restore(self, snap):
        a, pid, lk, pcs = snap
        self.alpha[...] = a
        self.piece_id[...] = pid
        self.locked[...] = lk
        self.pieces = [Piece(n, m, k, pr) for (n, m, k, pr) in pcs]
        return self

    # ── views ──────────────────────────────────────────────────────────────────
    @property
    def is_solid(self) -> np.ndarray:
        return self.alpha > 0.0

    def summary(self) -> str:
        solid = int(self.is_solid.sum())
        live = sum(1 for i in range(len(self.pieces)) if np.any(self.piece_id == i))
        return (f"Room {self.grid.NX} x {self.grid.NY} cells | "
                f"{solid} solid ({100 * solid / self.alpha.size:.1f}%) | {live} pieces")
