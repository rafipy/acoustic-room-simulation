"""scenes.py — room presets built with the Room engine.

`two_rooms` is the default demo scene; `tiny_home` is a small floor plan (concrete shell
with glass windows and a front door, drywall partitions making several rooms, furniture).
"""

from grid import RoomGrid
from room import Room


def two_rooms(grid: RoomGrid) -> Room:
    """Concrete shell, a wooden partition with a doorway, and a carpet sofa."""
    return (Room(grid)
            .add_border("Concrete", thickness=2)
            .add_rectangle("Wood", 4.0, 0.0, 4.15, 6.0, name="partition")
            .add_doorway(4.0, 2.5, 4.15, 3.5)
            .add_block("Carpet", 1.0, 1.0, 2.5, 2.0, name="sofa"))


def tiny_home(grid: RoomGrid) -> Room:
    """A small home: concrete shell with glass windows + a front door, drywall rooms, furniture."""
    r = Room(grid)
    W, H, t = grid.W, grid.H, 0.15

    # Outer shell (protected): concrete with glass window segments and a front-door gap.
    r.add_rectangle("Concrete", 0.0, 0.0, 1.2, t, name="wall S1", protected=True)
    r.add_rectangle("Glass",    1.2, 0.0, 2.2, t, name="window S", protected=True)
    r.add_rectangle("Concrete", 2.2, 0.0, 3.4, t, name="wall S2", protected=True)
    r.add_rectangle("Concrete", 4.4, 0.0, W,   t, name="wall S3", protected=True)   # door gap 3.4-4.4
    r.add_rectangle("Concrete", 0.0, H - t, 5.0, H, name="wall N1", protected=True)
    r.add_rectangle("Glass",    5.0, H - t, 6.2, H, name="window N", protected=True)
    r.add_rectangle("Concrete", 6.2, H - t, W,   H, name="wall N2", protected=True)
    r.add_rectangle("Concrete", 0.0, 0.0, t, H, name="wall W", protected=True)
    r.add_rectangle("Concrete", W - t, 0.0, W, 2.0, name="wall E1", protected=True)
    r.add_rectangle("Glass",    W - t, 2.0, W, 3.0, name="window E", protected=True)
    r.add_rectangle("Concrete", W - t, 3.0, W, H,   name="wall E2", protected=True)

    # Interior drywall partitions with doorway gaps -> several rooms.
    r.add_rectangle("Drywall", t, 3.6, 1.6, 3.6 + t, name="div H1")
    r.add_rectangle("Drywall", 2.6, 3.6, W - t, 3.6 + t, name="div H2")             # doorway 1.6-2.6
    r.add_rectangle("Drywall", 4.0, t, 4.0 + t, 1.4, name="div V1")
    r.add_rectangle("Drywall", 4.0, 2.4, 4.0 + t, 3.6, name="div V2")               # doorway 1.4-2.4

    # Furniture
    r.add_block("Carpet", 0.5, 0.5, 1.8, 1.4, name="sofa")
    r.add_block("Wood", 5.4, 4.3, 6.8, 5.2, name="table")
    return r
