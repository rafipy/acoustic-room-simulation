"""verify.py — consolidated checks for the acoustic FDTD project.

Run:  cd src && python verify.py
Covers the grid, the room builder, the scene presets, the FDTD solver, and a headless
run of the interactive app.  Exits non-zero if anything fails.
"""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # headless, for the visualiser check
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from config import C_SOUND, FREQ
from grid import RoomGrid
from utils import coord_to_cell, cell_to_coord, pressure_reflection
from room import Room
from physics_solver import WaveSolver
import scenes

SRC = (4.0, 3.0)


def _ok(name, passed):
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    return bool(passed)


def check_grid():
    g = RoomGrid()
    r, c = coord_to_cell(3.5, 2.0, g)
    x, y = cell_to_coord(r, c, g)
    roundtrip = abs(x - 3.5) <= g.dx and abs(y - 2.0) <= g.dy
    cfl = g.r < 1.0 / np.sqrt(2.0)
    refl = abs(float(pressure_reflection(0.0)) - 1.0) < 1e-9 and float(pressure_reflection(1.0)) < 1e-9
    return all([_ok("grid: CFL stable", cfl),
                _ok("grid: coord round-trip", roundtrip),
                _ok("grid: reflection limits (a=0 -> 1, a=1 -> 0)", refl)])


def check_room():
    g = RoomGrid()
    room = scenes.two_rooms(g)
    n0 = len(room.pieces)
    room.add_rectangle("Brick", 5.0, 1.0, 5.15, 4.0)
    built = len(room.pieces) == n0 + 1
    room.remove_piece(0)                                   # try to delete the shell (id 0)
    shell = float(room.alpha[0, 0]) > 0.0
    room.set_wall_material("Glass")
    grouped = all(p.material == "Glass" for p in room.pieces if p.kind == "wall")
    return all([_ok("room: build a piece", built),
                _ok("room: shell protected", shell),
                _ok("room: wall-group recolour", grouped)])


def check_scenes():
    th = scenes.tiny_home(RoomGrid())
    return _ok(f"scenes: tiny_home builds ({len(th.pieces)} pieces)",
               th.is_solid.any() and len(th.pieces) > 5)


def _window(solver, n):
    return float(np.mean(solver.run(n, record_energy=True)))


def check_solver():
    g = RoomGrid()
    results = []

    s = WaveSolver(g, Room(g).alpha); s.add_impulse(*SRC); s.run(1500)
    results.append(_ok("solver: stable over 1500 steps",
                       np.all(np.isfinite(s.field)) and float(np.max(np.abs(s.field))) < 100))

    rs = WaveSolver(g, Room(g).alpha); rs.add_impulse(*SRC); rs.run(400)
    e0, e1 = _window(rs, 400), _window(rs, 400)
    drift = abs(e1 - e0) / e0
    results.append(_ok(f"solver: rigid conserves energy (drift {drift*100:.2f}%)", drift < 0.10))

    ab = WaveSolver(g, Room(g).add_border("Acoustic Foam").alpha); ab.add_impulse(*SRC); ab.run(400)
    a0, a1 = _window(ab, 400), _window(ab, 400)
    results.append(_ok("solver: absorbing walls decay", a1 < 0.5 * a0))

    free = WaveSolver(g, Room(g).add_border("Open").alpha); free.add_impulse(*SRC, freq=FREQ)
    rsrc, csrc = coord_to_cell(*SRC, g)

    def front(sv):
        line = np.abs(sv.field[rsrc, csrc:])
        idx = np.where(line > 0.01 * float(np.max(np.abs(sv.field))))[0]
        return idx.max() * g.dx if len(idx) else 0.0

    free.run(45); r1, t1 = front(free), free.t
    free.run(30); r2, t2 = front(free), free.t
    speed = (r2 - r1) / (t2 - t1)
    results.append(_ok(f"solver: wavefront speed {speed:.0f} m/s ~ c={C_SOUND:.0f}",
                       0.85 < speed / C_SOUND < 1.10))

    def retained(mat):
        sv = WaveSolver(g, Room(g).add_border(mat).alpha); sv.add_impulse(*SRC); sv.run(400)
        e_a = _window(sv, 200); sv.run(1000); e_b = _window(sv, 200)
        return e_b / e_a if e_a > 0 else 0.0

    vals = [retained(m) for m in ("Concrete", "Wood", "Carpet", "Acoustic Foam")]
    mono = all(vals[i] >= vals[i + 1] - 1e-6 for i in range(len(vals) - 1))
    results.append(_ok(f"solver: absorption ladder {[round(v, 2) for v in vals]}", mono))
    return all(results)


def check_visualiser():
    from visualiser import Visualiser, SCALE
    import pygame

    app = Visualiser(headless=True)
    for _ in range(30):
        app.update()
    app.mode, app.tool, app.armed = "edit", "wall", "Brick"
    col = int(5.0 / app.grid.dx) * SCALE
    app.drag_from = (col, app.ch // 2)
    app._commit_drag((col, app.ch // 2 - 6 * SCALE))       # build a wall by dragging
    app.drag_from = None
    built = len(app.room.pieces)
    app._undo()                                            # undo it
    undone = len(app.room.pieces)
    app._erase_at(0, 0)                                    # shell is protected
    shell = float(app.room.alpha[0, 0]) > 0.0
    app._paint_at(*coord_to_cell(4.07, 5.0, app.grid))     # wall-group recolour
    app._damp(20); app._speed(4)
    for _ in range(10):
        app.update()
    app.draw()
    f = app.solver.field
    finite = bool(np.all(np.isfinite(f))) and float(np.max(np.abs(f))) < 100

    # energy history stays parallel (t vs value), the room auto-stops once it falls
    # silent, and a fresh clap re-arms (clears the auto-stop flag).
    app2 = Visualiser(headless=True)
    app2._damp(150)                                        # strong air damping -> quick decay
    for _ in range(3000):
        app2.update()
        if app2.auto_stopped:
            break
    parallel = len(app2.energy) == len(app2.energy_t) and len(app2.energy) > 0
    settled = app2.auto_stopped and app2.paused
    app2.mode = "source"
    ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": (app2.cw // 2, app2.ch // 2), "button": 1})
    app2.handle(ev)
    rearmed = not app2.auto_stopped

    pygame.quit()
    return all([_ok(f"app: build then undo ({built} -> {undone})", undone == built - 1),
                _ok("app: shell stays protected", shell),
                _ok("app: field finite & bounded", finite),
                _ok("app: energy history parallel (t vs value)", parallel),
                _ok("app: auto-stops when room falls silent", settled),
                _ok("app: fresh clap re-arms auto-stop", rearmed)])


if __name__ == "__main__":
    groups = [("Grid", check_grid), ("Room", check_room), ("Scenes", check_scenes),
              ("Solver", check_solver), ("Visualiser", check_visualiser)]
    passed = True
    for title, fn in groups:
        print(f"\n{title}")
        passed &= fn()
    print("\n" + ("ALL PASS" if passed else "SOME CHECKS FAILED"))
    sys.exit(0 if passed else 1)
