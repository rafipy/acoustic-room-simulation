"""
visualiser.py — interactive real-time view of the 2D acoustic FDTD simulation.

A clean white/black dashboard around the physics layer.  The room starts white (rest
pressure); a clap sends out a red compression wavefront that reflects, diffracts and
decays as it loses energy to the walls and a little air damping.  Walls/furniture are
coloured by material; controls live in the right sidebar; a live energy chart (with a
simulation clock) sits under the canvas.

Run:
    cd src
    python visualiser.py

Controls
--------
  MODE   Source : click the room to drop a clap (sound source).
         Edit   : pick a material swatch, then use a TOOL:
                    Paint  — click a wall (recolours ALL walls) or furniture (just it)
                    Wall   — drag to lay a wall; it auto-extends to connect (build rooms)
                    Block  — drag a furniture rectangle
                    Erase  — click a piece to delete it (outer shell is locked)
  [ / ] amplitude   , / . speed   Space pause   R clear   Ctrl+Z undo   Esc quit
"""

import os

import numpy as np
import pygame

from config import MATERIALS
from grid import RoomGrid
from physics_solver import WaveSolver
from render import field_to_rgb, to_surface, draw_line_chart
from scenes import two_rooms

# ── palette (white / black UI; walls coloured by material) ─────────────────────
WHITE = (255, 255, 255)
INK = (28, 28, 28)
SUBTLE = (140, 140, 140)
HOVER = (236, 236, 236)
LINE = (208, 208, 208)
ACCENT = (21, 101, 192)        # selection / drag preview outline (visible on white)

MATERIAL_COLORS = {
    "Concrete":      (150, 152, 156),
    "Brick":         (172, 76, 68),
    "Plaster":       (208, 196, 176),
    "Drywall":       (205, 203, 196),
    "Glass":         (120, 176, 196),
    "Wood":          (156, 108, 60),
    "Carpet":        (92, 124, 98),
    "Heavy Curtain": (126, 96, 150),
    "Acoustic Foam": (74, 74, 86),
    "Open":          (210, 210, 210),
}
DEFAULT_WALL_COLOR = (120, 120, 124)

SCALE = 6                      # screen pixels per grid cell
SIDEBAR_W = 300
CHART_H = 150
PAD = 14
FPS = 60
WALL_HALF = 1                  # Wall tool half-thickness in cells (-> 3 cells thick)
SNAP_CELLS = 6                 # a wall end only snaps to a solid within this many cells
DAMP_DEFAULT = 30.0            # air damping β so the room reverberates and falls silent
TOOLS = ["paint", "wall", "block", "erase"]

# Energy history + auto-stop (pause once the clap has fully died out).
ENERGY_MAX_SAMPLES = 6000      # cap on stored samples; halved when exceeded (keeps t=0)
STOP_ENERGY_FRAC = 1e-4        # "silent" = energy below this fraction of its peak
STOP_PEAK_FLOOR = 1e-2         # ...but only arm after a real clap (peak above this)
STOP_HOLD_FRAMES = 30          # stay quiet this many frames (~0.5 s) before pausing


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
        t = font.render(self.label, True, fg)
        surf.blit(t, t.get_rect(center=self.rect.center))


class Visualiser:
    def __init__(self, headless=False):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        self.grid = RoomGrid()
        self.room = two_rooms(self.grid)
        self.beta = DAMP_DEFAULT
        self.solver = WaveSolver(self.grid, self.room.alpha, beta=self.beta)

        self.mode = "source"
        self.tool = "paint"
        self.armed = next(iter(MATERIALS))
        self.selected = None
        self.amp = 1.0
        self.steps_per_frame = 4
        self.paused = False
        self.p_scale = 1e-4
        self.energy = []                                   # full Σu² history (parallel to energy_t)
        self.energy_t = []                                 # sim time (s) per energy sample
        self.energy_peak = 1e-9
        self.quiet_frames = 0                              # consecutive "silent" frames
        self.auto_stopped = False                          # paused because the room settled
        self.drag_from = None
        self.undo_stack = []
        self.msg = ""
        self.msg_frames = 0

        self.cw = self.grid.NX * SCALE
        self.ch = self.grid.NY * SCALE
        self._wall_rgb = None
        self._rebuild_wall_rgb()

        pygame.init()
        self.win_w = self.cw + SIDEBAR_W
        self.buttons = []
        self.swatches = []
        self.chart_rect = pygame.Rect(PAD, self.ch + PAD, self.cw - 2 * PAD, CHART_H - 2 * PAD)
        self._build_widgets()                              # sets self._content_bottom
        self.win_h = max(self.ch + CHART_H, self._content_bottom + PAD)

        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("Acoustic FDTD — interactive")
        self.clock = pygame.time.Clock()
        self.f_title = pygame.font.SysFont("segoeui,arial,dejavusans", 22, bold=True)
        self.f_head = pygame.font.SysFont("segoeui,arial,dejavusans", 13, bold=True)
        self.f_body = pygame.font.SysFont("segoeui,arial,dejavusans", 14)
        self.f_small = pygame.font.SysFont("segoeui,arial,dejavusans", 12)

        self.solver.add_impulse(2.0, 3.0, amp=self.amp)    # an opening clap

    # ── layout ────────────────────────────────────────────────────────────────
    def _build_widgets(self):
        x = self.cw + PAD
        w = SIDEBAR_W - 2 * PAD
        half = (w - 8) // 2
        quarter = (w - 18) // 4
        y = PAD + 50

        self._mode_hdr_y = y
        y += 18
        self.buttons.append(Button((x, y, half, 26), "Source", lambda: self._set_mode("source"),
                                   active=lambda: self.mode == "source"))
        self.buttons.append(Button((x + half + 8, y, half, 26), "Edit", lambda: self._set_mode("edit"),
                                   active=lambda: self.mode == "edit"))
        y += 34

        self._tool_hdr_y = y
        y += 18
        labels = {"paint": "Paint", "wall": "Wall", "block": "Block", "erase": "Erase"}
        for i, t in enumerate(TOOLS):
            self.buttons.append(Button((x + i * (quarter + 6), y, quarter, 26), labels[t],
                                       (lambda tt=t: self._set_tool(tt)),
                                       active=(lambda tt=t: self.mode == "edit" and self.tool == tt)))
        y += 38

        self._mat_hdr_y = y
        y += 18
        for name in MATERIALS:
            self.swatches.append((pygame.Rect(x, y, w, 21), name))
            y += 23
        y += 12

        self._sel_hdr_y = y
        y += 44

        self._amp_hdr_y = y
        y += 18
        self.buttons.append(Button((x, y, 30, 26), "-", lambda: self._amp(-0.25)))
        self.buttons.append(Button((x + w - 30, y, 30, 26), "+", lambda: self._amp(+0.25)))
        self._amp_val_rect = pygame.Rect(x + 34, y, w - 68, 26)
        y += 34

        self._spd_hdr_y = y
        y += 18
        self.buttons.append(Button((x, y, 30, 26), "-", lambda: self._speed(-2)))
        self.buttons.append(Button((x + w - 30, y, 30, 26), "+", lambda: self._speed(+2)))
        self._spd_val_rect = pygame.Rect(x + 34, y, w - 68, 26)
        y += 34

        self._dmp_hdr_y = y
        y += 18
        self.buttons.append(Button((x, y, 30, 26), "-", lambda: self._damp(-10)))
        self.buttons.append(Button((x + w - 30, y, 30, 26), "+", lambda: self._damp(+10)))
        self._dmp_val_rect = pygame.Rect(x + 34, y, w - 68, 26)
        y += 34

        self.buttons.append(Button((x, y, half, 26), "Pause", self._toggle_pause,
                                   active=lambda: self.paused))
        self.buttons.append(Button((x + half + 8, y, half, 26), "Clear", self._reset))
        y += 32
        self.buttons.append(Button((x, y, half, 26), "Undo", self._undo))
        self.buttons.append(Button((x + half + 8, y, half, 26), "Revert", self._revert))
        y += 34
        self._help_y = y
        self._sidebar_x = x
        self._sidebar_w = w
        self._content_bottom = y + 32

    # ── actions ─────────────────────────────────────────────────────────────────
    def _set_mode(self, m):
        self.mode = m

    def _set_tool(self, t):
        self.mode = "edit"
        self.tool = t

    def _arm(self, name):
        self.armed = name
        self.mode = "edit"
        if self.tool == "erase":
            self.tool = "paint"

    def _amp(self, d):
        self.amp = float(np.clip(self.amp + d, 0.25, 4.0))

    def _speed(self, d):
        self.steps_per_frame = int(np.clip(self.steps_per_frame + d, 0, 24))

    def _damp(self, d):
        self.beta = float(np.clip(self.beta + d, 0.0, 150.0))
        self.solver.set_beta(self.beta)

    def _toggle_pause(self):
        self.paused = not self.paused
        if not self.paused:                                # resuming re-arms auto-stop
            self.quiet_frames = 0
            self.auto_stopped = False

    def _reset(self):
        self.solver.reset()
        self.energy.clear()
        self.energy_t.clear()
        self.energy_peak = 1e-9
        self.p_scale = 1e-4
        self.quiet_frames = 0
        self.auto_stopped = False

    def _push_undo(self):
        self.undo_stack.append(self.room.snapshot())
        if len(self.undo_stack) > 40:
            self.undo_stack.pop(0)

    def _undo(self):
        if self.undo_stack:
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

    def _rebuild_wall_rgb(self):
        img = np.zeros((self.grid.NY, self.grid.NX, 3), dtype=np.uint8)
        for idx, piece in enumerate(self.room.pieces):
            mask = self.room.piece_id == idx
            if mask.any():
                img[mask] = material_color(piece.material)
        self._wall_rgb = img

    def _apply_geometry(self):
        self.solver.set_alpha(self.room.alpha)
        self._rebuild_wall_rgb()

    # ── coordinate helpers ──────────────────────────────────────────────────────
    def _cell_at(self, px, py):
        if 0 <= px < self.cw and 0 <= py < self.ch:
            return int((self.grid.NY - 1) - (py // SCALE)), int(px // SCALE)
        return None

    def _cell_metres(self, row, col):
        return col * self.grid.dx, row * self.grid.dy

    def _drag_rawbox(self, end_pix):
        if self.drag_from is None:
            return None
        a = self._cell_at(*self.drag_from)
        ex = min(max(end_pix[0], 0), self.cw - 1)
        ey = min(max(end_pix[1], 0), self.ch - 1)
        b = self._cell_at(ex, ey)
        if a is None or b is None:
            return None
        NY, NX = self.grid.NY, self.grid.NX
        ra, ca = a
        rb, cb = b
        if self.tool == "wall":
            if abs(cb - ca) >= abs(rb - ra):               # horizontal run
                return (max(0, ra - WALL_HALF), min(NY - 1, ra + WALL_HALF), min(ca, cb), max(ca, cb))
            return (min(ra, rb), max(ra, rb), max(0, ca - WALL_HALF), min(NX - 1, ca + WALL_HALF))
        return (min(ra, rb), max(ra, rb), min(ca, cb), max(ca, cb))   # block

    def _extend_wall(self, box):
        """Snap each end of a wall to a NEARBY solid (within SNAP_CELLS) so walls join
        up without jumping across the room.  Ends with no solid nearby stay as drawn."""
        solid = self.room.is_solid
        NY, NX = self.grid.NY, self.grid.NX
        r0, r1, c0, c1 = box

        def reach(start, step, span_lo, span_hi, axis_max, vertical):
            """Cells to extend toward a solid within SNAP_CELLS, else 0."""
            for k in range(1, SNAP_CELLS + 1):
                idx = start + step * k
                if idx < 0 or idx > axis_max:
                    return 0
                hit = (solid[idx, span_lo:span_hi + 1] if vertical
                       else solid[span_lo:span_hi + 1, idx]).any()
                if hit:
                    return k - 1            # stop adjacent to the solid
            return 0

        if (r1 - r0) >= (c1 - c0):                          # vertical wall
            r1 += reach(r1, +1, c0, c1, NY - 1, True)
            r0 -= reach(r0, -1, c0, c1, NY - 1, True)
        else:                                               # horizontal wall
            c1 += reach(c1, +1, r0, r1, NX - 1, False)
            c0 -= reach(c0, -1, r0, r1, NX - 1, False)
        return (r0, r1, c0, c1)

    def _current_box(self, end_pix):
        raw = self._drag_rawbox(end_pix)
        if raw is None:
            return None
        return self._extend_wall(raw) if self.tool == "wall" else raw

    def _cellbox_screen(self, box):
        r0, r1, c0, c1 = box
        return pygame.Rect(c0 * SCALE, (self.grid.NY - 1 - r1) * SCALE,
                           (c1 - c0 + 1) * SCALE, (r1 - r0 + 1) * SCALE)

    # ── event handling ───────────────────────────────────────────────────────────
    def handle(self, ev):
        if ev.type == pygame.QUIT:
            return False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_z and (ev.mod & pygame.KMOD_CTRL):
                self._undo()
            elif ev.key == pygame.K_ESCAPE:
                return False
            elif ev.key == pygame.K_SPACE:
                self._toggle_pause()
            elif ev.key == pygame.K_r:
                self._reset()
            elif ev.key == pygame.K_LEFTBRACKET:
                self._amp(-0.25)
            elif ev.key == pygame.K_RIGHTBRACKET:
                self._amp(+0.25)
            elif ev.key == pygame.K_COMMA:
                self._speed(-2)
            elif ev.key == pygame.K_PERIOD:
                self._speed(+2)

        elif ev.type == pygame.MOUSEBUTTONDOWN:
            px, py = ev.pos
            if px >= self.cw or py >= self.ch:
                if ev.button == 1:
                    self._click_panel(px, py)
                return True
            cell = self._cell_at(px, py)
            if cell is None:
                return True
            if self.mode == "source" and ev.button == 1:
                self.solver.add_impulse(*self._cell_metres(*cell), amp=self.amp)
                self.quiet_frames = 0                      # a fresh clap re-arms auto-stop
                self.auto_stopped = False
            elif self.mode == "edit":
                if ev.button == 3 or self.tool == "erase":
                    self._erase_at(*cell)
                elif self.tool == "paint":
                    self._paint_at(*cell)
                else:
                    self.drag_from = (px, py)

        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if self.drag_from is not None:
                self._commit_drag(ev.pos)
                self.drag_from = None
        return True

    def _click_panel(self, px, py):
        for b in self.buttons:
            if b.rect.collidepoint(px, py):
                b.on_click()
                return
        for rect, name in self.swatches:
            if rect.collidepoint(px, py):
                self._arm(name)
                return

    def _paint_at(self, row, col):
        pid = self.room.piece_at(row, col)
        if pid is None:
            self.selected = None
            return
        self._push_undo()
        if self.room.pieces[pid].kind == "wall":
            self.room.set_wall_material(self.armed)        # editing one wall edits all
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
        moved = abs(end_pix[0] - self.drag_from[0]) + abs(end_pix[1] - self.drag_from[1])
        if moved < SCALE:                                  # a click, not a drag
            return
        box = self._current_box(end_pix)
        if box is None:
            return
        self._push_undo()
        r0, r1, c0, c1 = box
        x0, y0 = self._cell_metres(r0, c0)
        x1, y1 = self._cell_metres(r1, c1)
        if self.tool == "wall":
            self.room.add_rectangle(self.armed, x0, y0, x1, y1,
                                    name=f"wall {len(self.room.pieces)}", kind="wall")
            self.room.set_wall_material(self.armed)        # all walls share one material
        else:
            self.room.add_block(self.armed, x0, y0, x1, y1, name=f"block {len(self.room.pieces)}")
        self.selected = len(self.room.pieces) - 1
        self._apply_geometry()

    # ── update ───────────────────────────────────────────────────────────────────
    def update(self):
        if self.msg_frames > 0:
            self.msg_frames -= 1
        if self.paused:
            return
        for _ in range(self.steps_per_frame):
            self.solver.step()
        peak = float(np.max(np.abs(self.solver.field)))
        self.p_scale = max(peak, self.p_scale * 0.97, 1e-4)
        e = self.solver.energy()
        self.energy.append(e)
        self.energy_t.append(self.solver.t)
        self.energy_peak = max(self.energy_peak, e)
        if len(self.energy) > ENERGY_MAX_SAMPLES:          # halve resolution, keep t=0
            self.energy = self.energy[::2]
            self.energy_t = self.energy_t[::2]
        self._check_auto_stop(e)

    def _check_auto_stop(self, e):
        """Pause once the clap has fully died out (canvas ~all white) and no source is
        still firing.  The threshold is relative to the peak energy, so a lossless rigid
        room (energy conserved, never decays) never trips it — only a genuinely decaying
        room does."""
        armed = (not self.solver.has_active_sources) and self.energy_peak > STOP_PEAK_FLOOR
        if armed and e < STOP_ENERGY_FRAC * self.energy_peak:
            self.quiet_frames += 1
        else:
            self.quiet_frames = 0
        if self.quiet_frames >= STOP_HOLD_FRAMES:
            self.paused = True
            self.auto_stopped = True
            self.quiet_frames = 0
            self._flash("room fell silent — Space to resume, click to clap")

    # ── drawing ──────────────────────────────────────────────────────────────────
    def draw(self):
        self.screen.fill(WHITE)
        mouse = pygame.mouse.get_pos()

        surf = to_surface(field_to_rgb(self.solver.field, self.room.alpha,
                                       self.p_scale, wall_rgb=self._wall_rgb))
        self.screen.blit(pygame.transform.scale(surf, (self.cw, self.ch)), (0, 0))
        self._draw_selection()
        self._draw_drag_preview(mouse)
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

    def _canvas_label(self):
        if self.mode == "source":
            msg = "SOURCE — click to clap"
        elif self.tool == "erase":
            msg = "EDIT · ERASE — click a piece"
        elif self.tool == "paint":
            msg = f"EDIT · PAINT — click a piece  (armed: {self.armed})"
        else:
            msg = f"EDIT · {self.tool.upper()} — drag to build  (armed: {self.armed})"
        chip = self.f_small.render("  " + msg + "  ", True, WHITE, INK)
        self.screen.blit(chip, (8, 8))
        if self.msg_frames > 0:
            note = self.f_small.render("  " + self.msg + "  ", True, WHITE, ACCENT)
            self.screen.blit(note, (8, 30))

    def _draw_chart(self):
        draw_line_chart(self.screen, self.chart_rect, self.energy, times=self.energy_t,
                        font=self.f_small, y_max=self.energy_peak)
        r = self.chart_rect
        cur = self.energy[-1] if self.energy else 0.0
        title = "Energy  Σu² (a.u.)" + ("   · SETTLED" if self.auto_stopped else "")
        self.screen.blit(self.f_head.render(title, True, INK), (r.x + 46, r.y + 1))
        self.screen.blit(self.f_small.render(
            f"now {cur:.3g}   peak {self.energy_peak:.3g}   t = {self.solver.t * 1e3:.0f} ms",
            True, SUBTLE), (r.right - 300, r.y + 1))

    def _section(self, text, y):
        self.screen.blit(self.f_head.render(text, True, INK), (self._sidebar_x, y))
        pygame.draw.line(self.screen, LINE, (self._sidebar_x, y - 8),
                         (self._sidebar_x + self._sidebar_w, y - 8))

    def _draw_sidebar(self, mouse):
        x, w = self._sidebar_x, self._sidebar_w
        self.screen.blit(self.f_title.render("Acoustic FDTD", True, INK), (x, PAD))
        self.screen.blit(self.f_small.render(
            f"{self.grid.NX}x{self.grid.NY} cells   dx={self.grid.dx*100:.1f} cm",
            True, SUBTLE), (x, PAD + 26))

        self._section("MODE", self._mode_hdr_y)
        self._section("TOOL  (Edit)", self._tool_hdr_y)
        self._section("MATERIAL  (click to arm)", self._mat_hdr_y)
        for rect, name in self.swatches:
            pygame.draw.rect(self.screen, material_color(name), (rect.x, rect.y + 2, 15, 15))
            pygame.draw.rect(self.screen, INK, (rect.x, rect.y + 2, 15, 15), 1)
            self.screen.blit(self.f_body.render(name, True, INK), (rect.x + 22, rect.y))
            self.screen.blit(self.f_small.render(f"α={MATERIALS[name]:.2f}", True, SUBTLE),
                             (rect.right - 44, rect.y + 2))
            if name == self.armed:
                pygame.draw.rect(self.screen, INK, rect.inflate(6, 4), 2)

        self._section("SELECTED", self._sel_hdr_y)
        sel = (self.room.pieces[self.selected].label()
               if self.selected is not None and self.selected < len(self.room.pieces)
               else "— (none)")
        self.screen.blit(self.f_body.render(sel, True, INK), (x, self._sel_hdr_y + 18))

        self._section("AMPLITUDE", self._amp_hdr_y)
        self._centered(self.f_body, f"{self.amp:.2f}", self._amp_val_rect)
        self._section("SPEED  (steps / frame)", self._spd_hdr_y)
        self._centered(self.f_body, f"{self.steps_per_frame}   (~{self.steps_per_frame * FPS}/s)",
                       self._spd_val_rect)
        self._section("DAMPING  β  (reverb decay)", self._dmp_hdr_y)
        self._centered(self.f_body, f"{self.beta:.0f}", self._dmp_val_rect)

        for b in self.buttons:
            b.draw(self.screen, self.f_body, mouse)

        hints = ["[ ] amp   , . speed   Ctrl+Z undo",
                 "Space pause   R clear   Esc quit"]
        for i, line in enumerate(hints):
            self.screen.blit(self.f_small.render(line, True, SUBTLE), (x, self._help_y + i * 16))

    def _centered(self, font, text, rect):
        t = font.render(text, True, INK)
        self.screen.blit(t, t.get_rect(center=rect.center))

    # ── loops ────────────────────────────────────────────────────────────────────
    def run(self):
        running = True
        while running:
            for ev in pygame.event.get():
                running = self.handle(ev) and running
            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()

def main():
    Visualiser().run()


if __name__ == "__main__":
    main()
