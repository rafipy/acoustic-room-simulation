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
    from visualiser import (Visualiser, SCALE, PRESSURE_GAMMA, PRESSURE_SCALE_FLOOR,
                            WAVE_VISUAL_GAIN,
                            CHART_H, PAD, TOOLS, SOURCE_DB_DEFAULT, SOURCE_DB_MIN,
                            SOURCE_DB_MAX, SPL_REF_PA)
    from render import field_to_rgb
    import pygame

    app = Visualiser(headless=True)
    for _ in range(30):
        app.update()
    scale_before = app.p_scale
    layout_bottom = (app.room_rect.top == 0 and app.chart_rect.top >= app.room_rect.bottom
                     and app.win_h >= app.ch + CHART_H)
    pressure_above_mode = app._pressure_hdr_y < app._mode_hdr_y
    sidebar_compact = (app.win_h <= app.ch + CHART_H
                       and app._content_bottom + PAD <= app.win_h)
    mapped = app._cell_at(*app.room_rect.center) is not None and app._cell_at(app.cw // 2, app.ch + 5) is None
    tool_order = TOOLS[0] == "block" and app.tool == "block"
    amp_units = f"{SOURCE_DB_DEFAULT:.1f} dB SPL" in app._amp_label() and "Pa" in app._amp_label()
    db_to_pa = abs(app.amp - SPL_REF_PA * (10.0 ** (SOURCE_DB_DEFAULT / 20.0))) < 1e-12
    pressure_db_labels = (app._pressure_db_label(SPL_REF_PA) == "0.0 dB SPL"
                          and app._pressure_level_label(SPL_REF_PA) == "0.0 dB"
                          and app._pressure_tick_label(PRESSURE_SCALE_FLOOR) == "14.0 dB"
                          and "e" not in app._pressure_db_label(PRESSURE_SCALE_FLOOR).lower()
                          and app._percent_label(1e-8) == "<0.001%")
    probe_uses_db = "dB" in app._probe_label(app.room_rect.center) and "Pa" not in app._probe_label(app.room_rect.center)
    app._amp(-999)
    db_min = app.source_db == SOURCE_DB_MIN
    app._amp(+999)
    db_max = app.source_db == SOURCE_DB_MAX
    app.source_db = SOURCE_DB_DEFAULT
    app.amp = app._db_to_pa(app.source_db)
    app._arm("Wood")
    selected_empty = app._selected_label() == "- (none)"
    selected_spacing = app._amp_hdr_y - 8 > app._sel_hdr_y + 18 + app.f_body.get_height()

    app.mode, app.tool, app.armed = "edit", "wall", "Brick"
    col = int(5.0 / app.grid.dx) * SCALE
    app.drag_from = (col, app.room_rect.centery)
    app._commit_drag((col, app.room_rect.centery - 6 * SCALE))       # build a wall by dragging
    app.drag_from = None
    built = len(app.room.pieces)
    app._undo()                                            # undo it
    undone = len(app.room.pieces)

    n0 = len(app.room.pieces)
    app.mode, app.tool, app.armed = "edit", "block", "Wood"
    click = app.room_rect.center
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": click, "button": 1}))
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, {"pos": click, "button": 1}))
    block_click = len(app.room.pieces) == n0 + 1 and app.room.pieces[-1].kind == "furniture"
    block_id = len(app.room.pieces) - 1
    ys, xs = np.where(app.room.piece_id == block_id)
    app.tool, app.armed = "paint", "Brick"
    if len(ys):
        app._paint_at(int(ys[0]), int(xs[0]))
    paint_furniture = block_click and app.room.pieces[block_id].material == "Brick"

    n1 = len(app.room.pieces)
    app.tool, app.armed = "block", "Carpet"
    app.drag_from = (app.room_rect.centerx - 8 * SCALE, app.room_rect.centery - 3 * SCALE)
    app._commit_drag((app.room_rect.centerx + 4 * SCALE, app.room_rect.centery + 5 * SCALE))
    app.drag_from = None
    block_drag = len(app.room.pieces) == n1 + 1 and app.room.pieces[-1].kind == "furniture"

    app._erase_at(0, 0)                                    # shell is protected
    shell = float(app.room.alpha[0, 0]) > 0.0
    app._paint_at(*coord_to_cell(4.07, 5.0, app.grid))     # wall-group recolour
    app._damp(20); app._speed(4)
    for _ in range(10):
        app.update()
    app.draw()
    before_preview = pygame.surfarray.array3d(app.screen).copy()
    app.mode, app.tool, app.drag_from = "edit", "block", None
    app._draw_drag_preview(app.room_rect.center)
    block_idle_preview = np.array_equal(before_preview, pygame.surfarray.array3d(app.screen))
    f = app.solver.field
    finite = bool(np.all(np.isfinite(f))) and float(np.max(np.abs(f))) < 100
    pressure_stats = (np.isfinite(app.pressure_max) and np.isfinite(app.pressure_rms)
                      and app.pressure_max >= 0.0 and app.pressure_rms >= 0.0)
    scale_monotone = app.p_scale >= scale_before
    sample_scale = max(app.p_scale, PRESSURE_SCALE_FLOOR)
    sample_field = np.array([[0.0, 0.25 * sample_scale]], dtype=float)
    sample_alpha = np.zeros_like(sample_field)
    true_rgb = field_to_rgb(sample_field, sample_alpha, sample_scale, gamma=PRESSURE_GAMMA)
    boosted_rgb = field_to_rgb(sample_field, sample_alpha, app._display_p_scale(), gamma=PRESSURE_GAMMA)
    visual_scale = abs(app._display_p_scale() - sample_scale / WAVE_VISUAL_GAIN) < 1e-12
    contrast_boost = int(boosted_rgb[0, 1, 1]) < int(true_rgb[0, 1, 1])
    parallel = len(app.energy) == len(app.energy_t) and len(app.energy) > 0

    app3 = Visualiser(headless=True)
    for _ in range(140):
        app3.update()
    app3.paused = True
    app3.mode = "source"
    app3.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                   {"pos": app3.room_rect.center, "button": 1}))
    pending = app3.pending_source is not None
    app3._toggle_pause()
    pending_fired = pending and app3.pending_source is None and app3.solver.has_active_sources

    # Even with strong damping, the room now keeps running until the user pauses
    # or clears it manually.
    app2 = Visualiser(headless=True)
    pieces_before_run = len(app2.room.pieces)
    app2._damp(150)                                        # strong air damping -> quick decay
    for _ in range(500):
        app2.update()
    no_auto_stop = (not app2.paused and len(app2.energy) > 0
                    and len(app2.energy_t) == len(app2.energy) and app2.solver.t > 0.0
                    and len(app2.room.pieces) == pieces_before_run)
    app2._reset()
    scale_resets = abs(app2.p_scale - PRESSURE_SCALE_FLOOR) < 1e-12

    pygame.quit()
    return all([_ok(f"app: build then undo ({built} -> {undone})", undone == built - 1),
                _ok("app: block click creates furniture", block_click),
                _ok("app: block drag creates furniture", block_drag),
                _ok("app: block is first tool with no idle preview", tool_order and block_idle_preview),
                _ok("app: paint recolours furniture", paint_furniture),
                _ok("app: shell stays protected", shell),
                _ok("app: bottom chart layout maps cells", layout_bottom and mapped),
                _ok("app: pressure panel above mode", pressure_above_mode),
                _ok("app: sidebar fits bottom-chart height", sidebar_compact),
                _ok("app: source label uses dB and Pa", amp_units and db_to_pa),
                _ok("app: source dB clamps to range", db_min and db_max),
                _ok("app: selected row omits armed material", selected_empty),
                _ok("app: selected row divider stays below text", selected_spacing),
                _ok("app: pressure scale labels use dB", pressure_db_labels),
                _ok("app: pressure probe uses dB", probe_uses_db),
                _ok("app: field finite & bounded", finite),
                _ok("app: pressure stats finite", pressure_stats),
                _ok("app: pressure scale is monotone per run", scale_monotone),
                _ok("app: wave render contrast boosted visually", visual_scale and contrast_boost),
                _ok("app: energy history parallel (t vs value)", parallel),
                _ok("app: paused source click fires on resume", pending_fired),
                _ok("app: no automatic pause", no_auto_stop),
                _ok("app: clear resets pressure scale", scale_resets)])


if __name__ == "__main__":
    groups = [("Grid", check_grid), ("Room", check_room), ("Scenes", check_scenes),
              ("Solver", check_solver), ("Visualiser", check_visualiser)]
    passed = True
    for title, fn in groups:
        print(f"\n{title}")
        passed &= fn()
    print("\n" + ("ALL PASS" if passed else "SOME CHECKS FAILED"))
    sys.exit(0 if passed else 1)
