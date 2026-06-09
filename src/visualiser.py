"""
visualiser.py - interactive real-time view of the 2D acoustic FDTD simulation.

The room shows scalar pressure as a red-on-white compression field.  Walls and
furniture are coloured by material, the energy chart sits below the room, and
the sidebar exposes source level, edit tools, damping, and pressure readouts.

Run:
    cd src
    python visualiser.py

Controls:
    Source mode: click to clap; while paused, click to arm a pending source.
    Edit mode: Block creates furniture, Paint recolours existing pieces,
               Wall drags wall segments, Erase deletes unlocked pieces.
    [ / ] source dB,  , / . speed,  Space pause,  R clear,
    Ctrl+Z undo,  Esc quit
"""

import os

import numpy as np
import pygame

from config import MATERIALS
from grid import RoomGrid
from physics_solver import WaveSolver
from render import field_to_rgb, to_surface, draw_line_chart, draw_pressure_legend
from scenes import two_rooms


WHITE = (255, 255, 255)
INK = (28, 28, 28)
SUBTLE = (140, 140, 140)
HOVER = (236, 236, 236)
LINE = (208, 208, 208)
ACCENT = (21, 101, 192)

MATERIAL_COLORS = {
    "Concrete": (150, 152, 156),
    "Brick": (172, 76, 68),
    "Plaster": (208, 196, 176),
    "Drywall": (205, 203, 196),
    "Glass": (120, 176, 196),
    "Wood": (156, 108, 60),
    "Carpet": (92, 124, 98),
    "Heavy Curtain": (126, 96, 150),
    "Acoustic Foam": (74, 74, 86),
    "Open": (210, 210, 210),
}
DEFAULT_WALL_COLOR = (120, 120, 124)

SCALE = 6
SIDEBAR_W = 300
CHART_H = 150
PAD = 14
FPS = 60

WALL_HALF = 1
SNAP_CELLS = 6
BLOCK_DEFAULT_W = 14
BLOCK_DEFAULT_H = 10

DAMP_DEFAULT = 30.0
TOOLS = ["block", "paint", "wall", "erase"]

ENERGY_MAX_SAMPLES = 6000

SPL_REF_PA = 20e-6
SOURCE_DB_DEFAULT = 60.0
SOURCE_DB_MIN = 40.0
SOURCE_DB_MAX = 120.0
SOURCE_DB_STEP = 5.0

PRESSURE_SCALE_FLOOR = 1e-4
PRESSURE_GAMMA = 0.6
WAVE_VISUAL_GAIN = 2.0


def material_color(material):
    if isinstance(material, str):
        return MATERIAL_COLORS.get(material, DEFAULT_WALL_COLOR)
    g = int(60 + 175 * (1.0 - float(material)))
    return (g, g, g)


class Button:
    def __init__(self, rect, label, on_click, active=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.active = active

    def draw(self, surf, font, mouse):
        on = self.active() if self.active else False
        hovered = self.rect.collidepoint(mouse)
        fill = INK if on else (HOVER if hovered else WHITE)
        fg = WHITE if on else INK
        pygame.draw.rect(surf, fill, self.rect)
        pygame.draw.rect(surf, INK, self.rect, 1)
        text = font.render(self.label, True, fg)
        surf.blit(text, text.get_rect(center=self.rect.center))


class Visualiser:
    def __init__(self, headless=False):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        self.grid = RoomGrid()
        self.room = two_rooms(self.grid)
        self.beta = DAMP_DEFAULT
        self.solver = WaveSolver(self.grid, self.room.alpha, beta=self.beta)

        self.mode = "source"
        self.tool = "block"
        self.armed = next(iter(MATERIALS))
        self.selected = None
        self.source_db = SOURCE_DB_DEFAULT
        self.amp = self._db_to_pa(self.source_db)
        self.steps_per_frame = 4
        self.paused = False
        self.auto_stopped = False
        self.pending_source = None

        self.p_scale = PRESSURE_SCALE_FLOOR
        self.pressure_max = 0.0
        self.pressure_rms = 0.0
        self.energy = []
        self.energy_t = []
        self.energy_peak = 1e-9

        self.drag_from = None
        self.undo_stack = []
        self.msg = ""
        self.msg_frames = 0

        self.cw = self.grid.NX * SCALE
        self.ch = self.grid.NY * SCALE
        self.room_rect = pygame.Rect(0, 0, self.cw, self.ch)
        self.chart_rect = pygame.Rect(PAD, self.ch + 8, self.cw - 2 * PAD, CHART_H - 2 * PAD - 22)
        self._wall_rgb = None
        self._rebuild_wall_rgb()

        pygame.init()
        self.win_w = self.cw + SIDEBAR_W
        self.buttons = []
        self.swatches = []
        self._build_widgets()
        self.win_h = max(self.ch + CHART_H, self._content_bottom + PAD)

        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("Acoustic FDTD - interactive")
        self.clock = pygame.time.Clock()
        self.f_title = pygame.font.SysFont("segoeui,arial,dejavusans", 22, bold=True)
        self.f_head = pygame.font.SysFont("segoeui,arial,dejavusans", 13, bold=True)
        self.f_body = pygame.font.SysFont("segoeui,arial,dejavusans", 14)
        self.f_small = pygame.font.SysFont("segoeui,arial,dejavusans", 12)

        self.solver.add_impulse(2.0, 3.0, amp=self.amp)

    # Layout
    def _build_widgets(self):
        x = self.cw + PAD
        w = SIDEBAR_W - 2 * PAD
        half = (w - 8) // 2
        quarter = (w - 18) // 4
        y = PAD + 58

        self._pressure_hdr_y = y
        y += 16
        self._pressure_y = y
        self._pressure_legend_rect = pygame.Rect(x, y + 28, w, 10)
        self._pressure_probe_y = y + 40
        y += 64

        self._mode_hdr_y = y
        y += 16
        self.buttons.append(Button((x, y, half, 24), "Source", lambda: self._set_mode("source"),
                                   active=lambda: self.mode == "source"))
        self.buttons.append(Button((x + half + 8, y, half, 24), "Edit", lambda: self._set_mode("edit"),
                                   active=lambda: self.mode == "edit"))
        y += 28

        self._tool_hdr_y = y
        y += 16
        labels = {"block": "Block", "paint": "Paint", "wall": "Wall", "erase": "Erase"}
        for i, tool in enumerate(TOOLS):
            self.buttons.append(Button((x + i * (quarter + 6), y, quarter, 24), labels[tool],
                                       (lambda tt=tool: self._set_tool(tt)),
                                       active=(lambda tt=tool: self.mode == "edit" and self.tool == tt)))
        y += 28

        self._mat_hdr_y = y
        y += 16
        for name in MATERIALS:
            self.swatches.append((pygame.Rect(x, y, w, 16), name))
            y += 16
        y += 8

        self._sel_hdr_y = y
        y += 46

        self._amp_hdr_y = y
        y += 16
        self.buttons.append(Button((x, y, 30, 24), "-", lambda: self._amp(-SOURCE_DB_STEP)))
        self.buttons.append(Button((x + w - 30, y, 30, 24), "+", lambda: self._amp(+SOURCE_DB_STEP)))
        self._amp_val_rect = pygame.Rect(x + 34, y, w - 68, 24)
        y += 30

        self._spd_hdr_y = y
        y += 16
        self.buttons.append(Button((x, y, 30, 24), "-", lambda: self._speed(-2)))
        self.buttons.append(Button((x + w - 30, y, 30, 24), "+", lambda: self._speed(+2)))
        self._spd_val_rect = pygame.Rect(x + 34, y, w - 68, 24)
        y += 28

        self._dmp_hdr_y = y
        y += 16
        self.buttons.append(Button((x, y, 30, 24), "-", lambda: self._damp(-10)))
        self.buttons.append(Button((x + w - 30, y, 30, 24), "+", lambda: self._damp(+10)))
        self._dmp_val_rect = pygame.Rect(x + 34, y, w - 68, 24)
        y += 30

        self.buttons.append(Button((x, y, half, 24), "Pause", self._toggle_pause,
                                   active=lambda: self.paused))
        self.buttons.append(Button((x + half + 8, y, half, 24), "Reset", self._reset))
        y += 26
        self.buttons.append(Button((x, y, half, 24), "Undo", self._undo))
        self.buttons.append(Button((x + half + 8, y, half, 24), "Revert", self._revert))
        y += 26

        self._help_y = y
        self._sidebar_x = x
        self._sidebar_w = w
        self._content_bottom = y + 18

    # Actions
    def _set_mode(self, mode):
        self.mode = mode

    def _set_tool(self, tool):
        self.mode = "edit"
        self.tool = tool

    def _arm(self, name):
        self.armed = name
        self.mode = "edit"
        if self.tool == "erase":
            self.tool = "block"

    def _db_to_pa(self, db):
        return float(SPL_REF_PA * (10.0 ** (float(db) / 20.0)))

    def _amp(self, delta_db):
        self.source_db = float(np.clip(self.source_db + delta_db, SOURCE_DB_MIN, SOURCE_DB_MAX))
        self.amp = self._db_to_pa(self.source_db)

    def _speed(self, delta):
        self.steps_per_frame = int(np.clip(self.steps_per_frame + delta, 0, 24))

    def _damp(self, delta):
        self.beta = float(np.clip(self.beta + delta, 0.0, 150.0))
        self.solver.set_beta(self.beta)

    def _toggle_pause(self):
        self.paused = not self.paused
        if not self.paused and self.pending_source is not None:
            pending = self.pending_source
            self.pending_source = None
            self._fire_source(pending)

    def _reset(self):
        self.solver.reset()
        if hasattr(self.solver, "_sources"):
            self.solver._sources.clear()
        self.pending_source = None
        self.energy.clear()
        self.energy_t.clear()
        self.energy_peak = 1e-9
        self.pressure_max = 0.0
        self.pressure_rms = 0.0
        self.p_scale = PRESSURE_SCALE_FLOOR
        self.auto_stopped = False

    def _push_undo(self):
        self.undo_stack.append(self.room.snapshot())
        if len(self.undo_stack) > 40:
            self.undo_stack.pop(0)

    def _undo(self):
        if not self.undo_stack:
            return
        self.room.restore(self.undo_stack.pop())
        self.selected = None
        self._apply_geometry()
        self._flash("undo")

    def _revert(self):
        self._push_undo()
        self.room = two_rooms(self.grid)
        self.selected = None
        self._apply_geometry()
        self._flash("reverted to the default room")

    def _flash(self, text):
        self.msg = text
        self.msg_frames = 150

    def _fire_source(self, cell):
        if cell is None:
            return
        self.solver.add_impulse(*self._cell_metres(*cell), amp=self.amp)
        self.auto_stopped = False

    def _rebuild_wall_rgb(self):
        image = np.zeros((self.grid.NY, self.grid.NX, 3), dtype=np.uint8)
        for idx, piece in enumerate(self.room.pieces):
            mask = self.room.piece_id == idx
            if mask.any():
                image[mask] = material_color(piece.material)
        self._wall_rgb = image

    def _apply_geometry(self):
        self.solver.set_alpha(self.room.alpha)
        self._rebuild_wall_rgb()

    # Coordinate helpers
    def _cell_at(self, px, py):
        if not self.room_rect.collidepoint(px, py):
            return None
        local_x = px - self.room_rect.x
        local_y = py - self.room_rect.y
        return int((self.grid.NY - 1) - (local_y // SCALE)), int(local_x // SCALE)

    def _cell_metres(self, row, col):
        return col * self.grid.dx, row * self.grid.dy

    def _cell_center_screen(self, row, col):
        return (self.room_rect.x + col * SCALE + SCALE // 2,
                self.room_rect.y + (self.grid.NY - 1 - row) * SCALE + SCALE // 2)

    def _drag_rawbox(self, end_pix):
        if self.drag_from is None:
            return None
        a = self._cell_at(*self.drag_from)
        ex = min(max(end_pix[0], self.room_rect.left), self.room_rect.right - 1)
        ey = min(max(end_pix[1], self.room_rect.top), self.room_rect.bottom - 1)
        b = self._cell_at(ex, ey)
        if a is None or b is None:
            return None

        ny, nx = self.grid.NY, self.grid.NX
        ra, ca = a
        rb, cb = b
        if self.tool == "wall":
            if abs(cb - ca) >= abs(rb - ra):
                return (max(0, ra - WALL_HALF), min(ny - 1, ra + WALL_HALF),
                        min(ca, cb), max(ca, cb))
            return (min(ra, rb), max(ra, rb), max(0, ca - WALL_HALF), min(nx - 1, ca + WALL_HALF))
        return min(ra, rb), max(ra, rb), min(ca, cb), max(ca, cb)

    def _extend_wall(self, box):
        solid = self.room.is_solid
        ny, nx = self.grid.NY, self.grid.NX
        r0, r1, c0, c1 = box

        def reach(start, step, span_lo, span_hi, axis_max, vertical):
            for k in range(1, SNAP_CELLS + 1):
                idx = start + step * k
                if idx < 0 or idx > axis_max:
                    return 0
                hit = (solid[idx, span_lo:span_hi + 1] if vertical
                       else solid[span_lo:span_hi + 1, idx]).any()
                if hit:
                    return k - 1
            return 0

        if (r1 - r0) >= (c1 - c0):
            r1 += reach(r1, +1, c0, c1, ny - 1, True)
            r0 -= reach(r0, -1, c0, c1, ny - 1, True)
        else:
            c1 += reach(c1, +1, r0, r1, nx - 1, False)
            c0 -= reach(c0, -1, r0, r1, nx - 1, False)
        return r0, r1, c0, c1

    def _current_box(self, end_pix):
        raw = self._drag_rawbox(end_pix)
        if raw is None:
            return None
        return self._extend_wall(raw) if self.tool == "wall" else raw

    def _default_block_box(self, row, col):
        ny, nx = self.grid.NY, self.grid.NX
        half_w = BLOCK_DEFAULT_W // 2
        half_h = BLOCK_DEFAULT_H // 2
        c0 = max(1, min(nx - 2, col - half_w))
        c1 = max(1, min(nx - 2, col + BLOCK_DEFAULT_W - half_w - 1))
        r0 = max(1, min(ny - 2, row - half_h))
        r1 = max(1, min(ny - 2, row + BLOCK_DEFAULT_H - half_h - 1))
        if c1 - c0 + 1 < BLOCK_DEFAULT_W:
            if c0 == 1:
                c1 = min(nx - 2, c0 + BLOCK_DEFAULT_W - 1)
            else:
                c0 = max(1, c1 - BLOCK_DEFAULT_W + 1)
        if r1 - r0 + 1 < BLOCK_DEFAULT_H:
            if r0 == 1:
                r1 = min(ny - 2, r0 + BLOCK_DEFAULT_H - 1)
            else:
                r0 = max(1, r1 - BLOCK_DEFAULT_H + 1)
        return r0, r1, c0, c1

    def _cellbox_screen(self, box):
        r0, r1, c0, c1 = box
        return pygame.Rect(self.room_rect.x + c0 * SCALE,
                           self.room_rect.y + (self.grid.NY - 1 - r1) * SCALE,
                           (c1 - c0 + 1) * SCALE,
                           (r1 - r0 + 1) * SCALE)

    # Event handling
    def handle(self, event):
        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_z and (event.mod & pygame.KMOD_CTRL):
                self._undo()
            elif event.key == pygame.K_ESCAPE:
                return False
            elif event.key == pygame.K_SPACE:
                self._toggle_pause()
            elif event.key == pygame.K_r:
                self._reset()
            elif event.key == pygame.K_LEFTBRACKET:
                self._amp(-SOURCE_DB_STEP)
            elif event.key == pygame.K_RIGHTBRACKET:
                self._amp(+SOURCE_DB_STEP)
            elif event.key == pygame.K_COMMA:
                self._speed(-2)
            elif event.key == pygame.K_PERIOD:
                self._speed(+2)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            px, py = event.pos
            if not self.room_rect.collidepoint(px, py):
                if event.button == 1:
                    self._click_panel(px, py)
                return True

            cell = self._cell_at(px, py)
            if cell is None:
                return True

            if self.mode == "source" and event.button == 1:
                if self.paused:
                    self.pending_source = cell
                    self._flash("source armed - Space to fire")
                else:
                    self._fire_source(cell)
            elif self.mode == "edit":
                if event.button == 3 or self.tool == "erase":
                    self._erase_at(*cell)
                elif event.button == 1 and self.tool == "paint":
                    self._paint_at(*cell)
                elif event.button == 1 and self.tool in ("block", "wall"):
                    self.drag_from = (px, py)

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.drag_from is not None:
                self._commit_drag(event.pos)
                self.drag_from = None

        return True

    def _click_panel(self, px, py):
        for button in self.buttons:
            if button.rect.collidepoint(px, py):
                button.on_click()
                return
        for rect, name in self.swatches:
            if rect.collidepoint(px, py):
                self._arm(name)
                return

    def _paint_at(self, row, col):
        pid = self.room.piece_at(row, col)
        if pid is None:
            self.selected = None
            self._flash("Paint recolours existing walls or blocks")
            return

        self._push_undo()
        if self.room.pieces[pid].kind == "wall":
            self.room.set_wall_material(self.armed)
        else:
            self.room.set_material(pid, self.armed)
        self.selected = pid
        self._apply_geometry()

    def _erase_at(self, row, col):
        pid = self.room.piece_at(row, col)
        if pid is None:
            return
        if self.room.pieces[pid].protected:
            self._flash("outer wall is locked")
            return

        self._push_undo()
        self.room.remove_piece(pid)
        if self.selected == pid:
            self.selected = None
        self._apply_geometry()

    def _commit_drag(self, end_pix):
        if self.drag_from is None:
            return
        moved = abs(end_pix[0] - self.drag_from[0]) + abs(end_pix[1] - self.drag_from[1])
        if moved < SCALE:
            if self.tool == "block":
                cell = self._cell_at(*self.drag_from)
                if cell is not None:
                    self._add_box(self._default_block_box(*cell), furniture=True)
            return

        box = self._current_box(end_pix)
        if box is None:
            return
        self._add_box(box, furniture=(self.tool == "block"))

    def _add_box(self, box, furniture):
        self._push_undo()
        r0, r1, c0, c1 = box
        x0, y0 = self._cell_metres(r0, c0)
        x1, y1 = self._cell_metres(r1, c1)
        if furniture:
            self.room.add_block(self.armed, x0, y0, x1, y1, name=f"block {len(self.room.pieces)}")
        else:
            self.room.add_rectangle(self.armed, x0, y0, x1, y1,
                                    name=f"wall {len(self.room.pieces)}", kind="wall")
            self.room.set_wall_material(self.armed)
        self.selected = len(self.room.pieces) - 1
        self._apply_geometry()

    # Update
    def update(self):
        if self.msg_frames > 0:
            self.msg_frames -= 1
        if self.paused:
            return

        for _ in range(self.steps_per_frame):
            self.solver.step()
        self._update_pressure_stats()

        energy = self.solver.energy()
        self.energy.append(energy)
        self.energy_t.append(self.solver.t)
        self.energy_peak = max(self.energy_peak, energy)
        if len(self.energy) > ENERGY_MAX_SAMPLES:
            self.energy = self.energy[::2]
            self.energy_t = self.energy_t[::2]

    def _update_pressure_stats(self):
        field = self.solver.field
        air = self.room.alpha <= 0.0
        if air.any():
            samples = field[air]
        else:
            samples = field.ravel()
        self.pressure_max = float(np.max(np.abs(samples))) if samples.size else 0.0
        self.pressure_rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
        self.p_scale = max(self.p_scale, self.pressure_max, PRESSURE_SCALE_FLOOR)

    def _display_p_scale(self):
        return max(self.p_scale, PRESSURE_SCALE_FLOOR) / WAVE_VISUAL_GAIN

    # Formatting helpers
    def _pa_label(self, value):
        value = float(abs(value))
        if value >= 10.0:
            return f"{value:.0f} Pa"
        if value >= 1.0:
            return f"{value:.2f} Pa"
        if value >= 0.01:
            return f"{value:.3f} Pa"
        if value >= 0.001:
            return f"{value:.4f} Pa"
        return f"{value:.4g} Pa"

    def _amp_label(self):
        return f"{self.source_db:.1f} dB SPL ({self._pa_label(self.amp)})"

    def _pressure_db_label(self, pressure):
        p = max(abs(float(pressure)), 1e-12)
        db = 20.0 * np.log10(p / SPL_REF_PA)
        return f"{db:.1f} dB SPL"

    def _pressure_level_label(self, pressure):
        return self._pressure_db_label(pressure).replace(" SPL", "")

    def _pressure_tick_label(self, pressure):
        if pressure <= 0.0:
            return "0"
        return self._pressure_level_label(pressure)

    def _percent_label(self, fraction):
        pct = float(fraction) * 100.0
        if 0.0 < pct < 0.001:
            return "<0.001%"
        return f"{pct:.3g}%"

    def _selected_label(self):
        if self.selected is not None and self.selected < len(self.room.pieces):
            piece = self.room.pieces[self.selected]
            material = piece.material
            return f"{piece.name} ({material})"
        return "- (none)"

    # Drawing
    def draw(self):
        self.screen.fill(WHITE)
        mouse = pygame.mouse.get_pos()

        rgb = field_to_rgb(self.solver.field, self.room.alpha, self._display_p_scale(),
                           wall_rgb=self._wall_rgb, gamma=PRESSURE_GAMMA)
        surf = to_surface(rgb)
        self.screen.blit(pygame.transform.scale(surf, (self.cw, self.ch)), self.room_rect.topleft)
        self._draw_selection()
        self._draw_drag_preview(mouse)
        self._draw_source_preview(mouse)
        self._canvas_label()

        pygame.draw.line(self.screen, LINE, (self.cw, 0), (self.cw, self.win_h))
        pygame.draw.line(self.screen, LINE, (0, self.ch), (self.cw, self.ch))

        self._draw_chart()
        self._draw_sidebar(mouse)
        pygame.display.flip()

    def _draw_selection(self):
        if self.selected is None:
            return
        ys, xs = np.where(self.room.piece_id == self.selected)
        if len(xs):
            pygame.draw.rect(self.screen, ACCENT,
                             self._cellbox_screen((ys.min(), ys.max(), xs.min(), xs.max())), 2)

    def _draw_drag_preview(self, mouse):
        if self.drag_from is None:
            return
        box = self._current_box(mouse)
        if box is not None:
            pygame.draw.rect(self.screen, ACCENT, self._cellbox_screen(box), 2)

    def _draw_source_preview(self, mouse):
        if self.mode != "source":
            return
        cells = []
        hover = self._cell_at(*mouse)
        if hover is not None:
            cells.append((hover, ACCENT))
        if self.pending_source is not None:
            cells.append((self.pending_source, INK))
        for cell, color in cells:
            cx, cy = self._cell_center_screen(*cell)
            pygame.draw.circle(self.screen, color, (cx, cy), 12, 2)
            pygame.draw.line(self.screen, color, (cx - 18, cy), (cx + 18, cy), 1)
            pygame.draw.line(self.screen, color, (cx, cy - 18), (cx, cy + 18), 1)

    def _canvas_label(self):
        if self.mode == "source":
            if self.paused:
                msg = "SOURCE - click to arm, Space to fire"
            else:
                msg = "SOURCE - click to clap"
        elif self.tool == "erase":
            msg = "EDIT - ERASE - click a piece"
        elif self.tool == "paint":
            msg = f"EDIT - PAINT - click a piece (armed: {self.armed})"
        elif self.tool == "wall":
            msg = f"EDIT - WALL - drag to build (armed: {self.armed})"
        else:
            msg = f"EDIT - BLOCK - click or drag furniture (armed: {self.armed})"

        chip = self.f_small.render("  " + msg + "  ", True, WHITE, INK)
        self.screen.blit(chip, (8, 8))
        if self.msg_frames > 0:
            note = self.f_small.render("  " + self.msg + "  ", True, WHITE, ACCENT)
            self.screen.blit(note, (8, 30))

    def _draw_chart(self):
        draw_line_chart(self.screen, self.chart_rect, self.energy, times=self.energy_t,
                        font=self.f_small, y_max=self.energy_peak)
        r = self.chart_rect
        current = self.energy[-1] if self.energy else 0.0
        title = "Energy  sum u^2 (a.u.)"
        label_y = r.bottom + 4
        self.screen.blit(self.f_head.render(title, True, INK), (r.x + 46, label_y))
        self.screen.blit(self.f_small.render(
            f"now {current:.3g}   peak {self.energy_peak:.3g}   t = {self.solver.t * 1e3:.0f} ms",
            True, SUBTLE), (r.right - 300, label_y + 1))

    def _section(self, text, y):
        self.screen.blit(self.f_head.render(text, True, INK), (self._sidebar_x, y))
        pygame.draw.line(self.screen, LINE, (self._sidebar_x, y - 6),
                         (self._sidebar_x + self._sidebar_w, y - 6))

    def _draw_sidebar(self, mouse):
        x, w = self._sidebar_x, self._sidebar_w
        title_y = PAD + 8
        self.screen.blit(self.f_title.render("Acoustic FDTD", True, INK), (x, title_y))
        self.screen.blit(self.f_small.render(
            f"{self.grid.NX}x{self.grid.NY} cells   dx={self.grid.dx * 100:.1f} cm",
            True, SUBTLE), (x, title_y + 26))

        self._draw_pressure_panel(mouse)
        self._section("MODE", self._mode_hdr_y)
        self._section("TOOL  (Edit)", self._tool_hdr_y)
        self._section("MATERIAL  (click to arm)", self._mat_hdr_y)

        for rect, name in self.swatches:
            pygame.draw.rect(self.screen, material_color(name), (rect.x, rect.y + 2, 13, 13))
            pygame.draw.rect(self.screen, INK, (rect.x, rect.y + 2, 13, 13), 1)
            self.screen.blit(self.f_small.render(name, True, INK), (rect.x + 20, rect.y + 1))
            self.screen.blit(self.f_small.render(f"α={MATERIALS[name]:.2f}", True, SUBTLE),
                             (rect.right - 44, rect.y + 1))
            if name == self.armed:
                pygame.draw.rect(self.screen, INK, rect, 2)

        self._section("SELECTED", self._sel_hdr_y)
        self.screen.blit(self.f_body.render(self._selected_label(), True, INK),
                         (x, self._sel_hdr_y + 17))

        self._section("SOURCE LEVEL", self._amp_hdr_y)
        self._centered(self.f_body, self._amp_label(), self._amp_val_rect)

        self._section("SPEED  (steps / frame)", self._spd_hdr_y)
        self._centered(self.f_body, f"{self.steps_per_frame}   (~{self.steps_per_frame * FPS}/s)",
                       self._spd_val_rect)

        self._section("DAMPING  β  (reverb decay)", self._dmp_hdr_y)
        self._centered(self.f_body, f"{self.beta:.0f}", self._dmp_val_rect)

        for button in self.buttons:
            button.draw(self.screen, self.f_body, mouse)

        hints = ["[ ] source dB   , . speed   Ctrl+Z undo",
                 "Space pause   R reset   Esc quit"]
        for i, line in enumerate(hints):
            self.screen.blit(self.f_small.render(line, True, SUBTLE), (x, self._help_y + i * 14))

    def _draw_pressure_panel(self, mouse):
        x, w = self._sidebar_x, self._sidebar_w
        self._section("PRESSURE", self._pressure_hdr_y)

        current = self.energy[-1] if self.energy else 0.0
        frac = current / self.energy_peak if self.energy_peak > 0 else 0.0
        left = f"max {self._pressure_level_label(self.pressure_max)}   RMS {self._pressure_level_label(self.pressure_rms)}"
        right = f"energy {self._percent_label(frac)}   scale {self._pressure_level_label(self.p_scale)}"
        self.screen.blit(self.f_small.render(left, True, SUBTLE), (x, self._pressure_y))
        self.screen.blit(self.f_small.render(right, True, SUBTLE), (x, self._pressure_y + 14))

        draw_pressure_legend(self.screen, self._pressure_legend_rect, None,
                             max(self.p_scale, PRESSURE_SCALE_FLOOR),
                             gamma=PRESSURE_GAMMA)

        probe = self._probe_label(mouse)
        self.screen.blit(self.f_small.render(probe, True, SUBTLE), (x, self._pressure_probe_y))

    def _probe_label(self, mouse):
        cell = self._cell_at(*mouse)
        if cell is None:
            return "probe: move over room"
        row, col = cell
        u = float(self.solver.field[row, col])
        red = np.clip(max(u, 0.0) / max(self._display_p_scale(), 1e-12), 0.0, 1.0) ** PRESSURE_GAMMA
        return f"cell {col},{row}  level {self._pressure_level_label(abs(u))}  red {red * 100:.0f}%"

    def _centered(self, font, text, rect):
        rendered = font.render(text, True, INK)
        self.screen.blit(rendered, rendered.get_rect(center=rect.center))

    # Main loop
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                running = self.handle(event) and running
            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()


def main():
    Visualiser().run()


if __name__ == "__main__":
    main()
