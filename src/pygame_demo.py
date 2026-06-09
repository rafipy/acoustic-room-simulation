"""Interactive acoustic simulator driven by the floor plan alpha map.

Flow:
1) Setup mode in Pygame: adjust parameters and click to place the sound source.
2) Compile mode (triggered by Enter): build solver state from the setup choices.
3) Run mode: animate the pressure field produced by the solver.

The room geometry always comes from floor_plan.create_floor_plan().
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import pygame

from config import (
    C_SOUND,
    FREQ,
    FLOOR_PLAN_CELL_SIZE,
    FLOOR_PLAN_DX,
    FLOOR_PLAN_GRID_PX_H,
    FLOOR_PLAN_GRID_PX_W,
    FLOOR_PLAN_MARGIN_LEFT,
    FLOOR_PLAN_MARGIN_TOP,
    FLOOR_PLAN_NX,
    FLOOR_PLAN_NY,
    FLOOR_PLAN_ROOM_HEIGHT_M,
    FLOOR_PLAN_ROOM_WIDTH_M,
    FLOOR_PLAN_SCREEN_H,
    FLOOR_PLAN_SCREEN_W,
    MATERIALS,
)
from floor_plan import create_floor_plan
from grid import RoomGrid
from physics_solver import WaveSolver


AIR_COLOR = (15, 18, 22)
PANEL_COLOR = (36, 39, 46)
TEXT_COLOR = (235, 235, 235)
HIGHLIGHT_COLOR = (90, 166, 255)
SOURCE_COLOR = (255, 236, 121)
EXTRA_RIGHT_PANEL_PX = 520

WALL_COLOR_ANCHORS = [
    (0.00, (255, 255, 204)),
    (0.20, (254, 217, 118)),
    (0.40, (253, 141, 60)),
    (0.60, (252, 78, 42)),
    (0.80, (227, 26, 28)),
    (1.00, (128, 0, 38)),
]


@dataclass
class SimParams:
    amp: float = 1.0
    freq_hz: float = FREQ / 2.0
    beta: float = 0.0
    steps_per_frame: int = 3
    fps: int = 60
    reclap_every: int = 1200


@dataclass
class CompiledSession:
    grid: RoomGrid
    alpha: np.ndarray
    solver: WaveSolver
    source_row: int
    source_col: int
    p_scale: float = 1e-4


def clamp_float(v: float, low: float, high: float) -> float:
    return max(low, min(high, v))


def clamp_int(v: int, low: int, high: int) -> int:
    return max(low, min(high, v))


def make_grid_for_floor_plan(beta: float) -> RoomGrid:
    # Choose ppw so RoomGrid.dx matches FLOOR_PLAN_DX exactly.
    wavelength = C_SOUND / FREQ
    ppw = wavelength / FLOOR_PLAN_DX
    return RoomGrid(
        width=FLOOR_PLAN_ROOM_WIDTH_M,
        height=FLOOR_PLAN_ROOM_HEIGHT_M,
        ppw=ppw,
        beta=beta,
    )


def _interpolate_color(anchors: list[tuple[float, tuple[int, int, int]]], t: float) -> tuple[int, int, int]:
    t = float(np.clip(t, 0.0, 1.0))
    for i in range(len(anchors) - 1):
        t0, c0 = anchors[i]
        t1, c1 = anchors[i + 1]
        if t <= t1:
            s = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return (
                int(c0[0] + s * (c1[0] - c0[0])),
                int(c0[1] + s * (c1[1] - c0[1])),
                int(c0[2] + s * (c1[2] - c0[2])),
            )
    return anchors[-1][1]


def wave_color_from_normalized(v: float) -> tuple[int, int, int]:
    pos = max(v, 0.0)
    neg = max(-v, 0.0)
    return (int(20 + 235 * pos), 20, int(20 + 235 * neg))


def wall_color_from_alpha(alpha: float) -> tuple[int, int, int]:
    return _interpolate_color(WALL_COLOR_ANCHORS, alpha)


def compile_session(params: SimParams, source_row: int, source_col: int) -> CompiledSession:
    grid = make_grid_for_floor_plan(params.beta)
    alpha = create_floor_plan().astype(np.float64)

    if alpha.shape != (grid.NY, grid.NX):
        raise ValueError(
            f"Floor plan shape {alpha.shape} does not match solver grid {(grid.NY, grid.NX)}"
        )

    solver = WaveSolver(grid, alpha)
    solver.add_impulse(source_col * grid.dx, source_row * grid.dy, amp=params.amp, freq=params.freq_hz)
    return CompiledSession(
        grid=grid,
        alpha=alpha,
        solver=solver,
        source_row=source_row,
        source_col=source_col,
    )


def pressure_to_rgb(field: np.ndarray, alpha: np.ndarray, p_scale: float) -> np.ndarray:
    val = field / max(p_scale, 1e-9)
    pos = np.clip(val, 0.0, 1.0)
    neg = np.clip(-val, 0.0, 1.0)

    rgb = np.empty(field.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = (20 + 235 * pos).astype(np.uint8)
    rgb[..., 1] = 20
    rgb[..., 2] = (20 + 235 * neg).astype(np.uint8)

    solid = alpha > 0.0
    if np.any(solid):
        solid_alpha = np.clip(alpha[solid], 0.0, 1.0)
        solid_colors = np.array([wall_color_from_alpha(a) for a in solid_alpha], dtype=np.uint8)
        rgb[solid] = solid_colors
    return rgb


def alpha_to_rgb(alpha: np.ndarray) -> np.ndarray:
    rgb = np.empty(alpha.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = AIR_COLOR[0]
    rgb[..., 1] = AIR_COLOR[1]
    rgb[..., 2] = AIR_COLOR[2]

    solid = alpha > 0.0
    if np.any(solid):
        solid_alpha = np.clip(alpha[solid], 0.0, 1.0)
        solid_colors = np.array([wall_color_from_alpha(a) for a in solid_alpha], dtype=np.uint8)
        rgb[solid] = solid_colors
    return rgb


def draw_legends(
    screen: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    legend_x: int,
    panel_y: int,
    legend_w: int,
    p_scale: float,
) -> None:
    bar_h = 160
    bar_w = 18
    left_x = legend_x + 8
    top_y = panel_y + 16

    # Wave pressure legend: top=positive(red), bottom=negative(blue)
    for i in range(bar_h):
        t = 1.0 - i / max(1, bar_h - 1)
        v = (2.0 * t) - 1.0
        color = wave_color_from_normalized(v)
        pygame.draw.line(screen, color, (left_x, top_y + i), (left_x + bar_w - 1, top_y + i))
    pygame.draw.rect(screen, (0, 0, 0), (left_x, top_y, bar_w, bar_h), 1)

    wave_label = small_font.render("Wave", True, TEXT_COLOR)
    screen.blit(wave_label, (left_x - 2, top_y - 16))

    for frac, text in [(1.0, f"+{p_scale:.1e}"), (0.5, "0"), (0.0, f"-{p_scale:.1e}")]:
        y = top_y + int((1.0 - frac) * bar_h)
        pygame.draw.line(screen, TEXT_COLOR, (left_x + bar_w, y), (left_x + bar_w + 4, y), 1)
        lbl = small_font.render(text, True, TEXT_COLOR)
        screen.blit(lbl, (left_x + bar_w + 8, y - lbl.get_height() // 2))

    # Material absorption legend
    mat_x = left_x + 76
    for i in range(bar_h):
        t = 1.0 - i / max(1, bar_h - 1)
        color = wall_color_from_alpha(t)
        pygame.draw.line(screen, color, (mat_x, top_y + i), (mat_x + bar_w - 1, top_y + i))
    pygame.draw.rect(screen, (0, 0, 0), (mat_x, top_y, bar_w, bar_h), 1)

    mat_label = small_font.render("Walls (alpha)", True, TEXT_COLOR)
    screen.blit(mat_label, (mat_x - 8, top_y - 16))
    for frac, text in [(1.0, "1.00"), (0.5, "0.50"), (0.0, "0.00")]:
        y = top_y + int((1.0 - frac) * bar_h)
        pygame.draw.line(screen, TEXT_COLOR, (mat_x + bar_w, y), (mat_x + bar_w + 4, y), 1)
        lbl = small_font.render(text, True, TEXT_COLOR)
        screen.blit(lbl, (mat_x + bar_w + 8, y - lbl.get_height() // 2))

    legend_title = small_font.render("Legends", True, TEXT_COLOR)
    screen.blit(legend_title, (legend_x + max(0, (legend_w - legend_title.get_width()) // 2), panel_y + 4))


def wrap_text(font: pygame.font.Font, text: str, max_width: int) -> list[str]:
    if not text:
        return [""]

    words = text.split()
    if not words:
        return [text]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.size(candidate)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def to_surface(rgb: np.ndarray) -> pygame.Surface:
    flipped = rgb[::-1]
    arr = np.ascontiguousarray(np.transpose(flipped, (1, 0, 2)))
    return pygame.surfarray.make_surface(arr)


def screen_to_cell(px: int, py: int) -> tuple[int, int] | None:
    gx = px - FLOOR_PLAN_MARGIN_LEFT
    gy = py - FLOOR_PLAN_MARGIN_TOP
    if gx < 0 or gy < 0 or gx >= FLOOR_PLAN_GRID_PX_W or gy >= FLOOR_PLAN_GRID_PX_H:
        return None
    col = gx // FLOOR_PLAN_CELL_SIZE
    row_from_top = gy // FLOOR_PLAN_CELL_SIZE
    row = (FLOOR_PLAN_NY - 1) - row_from_top
    return int(row), int(col)


def cell_to_screen_center(row: int, col: int) -> tuple[int, int]:
    px = FLOOR_PLAN_MARGIN_LEFT + col * FLOOR_PLAN_CELL_SIZE + FLOOR_PLAN_CELL_SIZE // 2
    py_top = FLOOR_PLAN_MARGIN_TOP + (FLOOR_PLAN_NY - 1 - row) * FLOOR_PLAN_CELL_SIZE
    py = py_top + FLOOR_PLAN_CELL_SIZE // 2
    return px, py


def draw_panel(
    screen: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    params: SimParams,
    source_row: int | None,
    source_col: int | None,
    status: str,
    selected_index: int,
    mode_name: str,
    steps: int,
    sim_time_ms: float,
    alpha: np.ndarray,
    p_scale: float,
) -> None:
    panel_x = FLOOR_PLAN_MARGIN_LEFT + FLOOR_PLAN_GRID_PX_W + 20
    panel_y = FLOOR_PLAN_MARGIN_TOP
    panel_w = screen.get_width() - panel_x - 12
    panel_h = FLOOR_PLAN_GRID_PX_H
    pygame.draw.rect(screen, PANEL_COLOR, (panel_x, panel_y, panel_w, panel_h), border_radius=8)

    legend_col_w = 210
    legend_x = panel_x + panel_w - legend_col_w - 10
    text_left = panel_x + 10
    text_right = legend_x - 10

    pygame.draw.line(screen, (70, 74, 84), (legend_x - 6, panel_y + 8), (legend_x - 6, panel_y + panel_h - 8), 1)
    draw_legends(screen, font, small_font, legend_x, panel_y, legend_col_w, p_scale)

    lines = [
        f"Mode: {mode_name}",
        "",
        "Parameter controls:",
        "UP/DOWN: select field",
        "LEFT/RIGHT: adjust value",
        "",
        f"1) amp          : {params.amp:.2f}",
        f"2) freq_hz      : {params.freq_hz:.1f}",
        f"3) beta         : {params.beta:.3f}",
        f"4) steps/frame  : {params.steps_per_frame}",
        f"5) fps          : {params.fps}",
        f"6) reclap_every : {params.reclap_every}",
        "",
        "Mouse left click in grid:",
        "Set source location",
        "",
        f"source cell: {source_row, source_col}" if source_row is not None else "source cell: not set",
        f"steps: {steps}",
        f"sim t: {sim_time_ms:.2f} ms",
        "",
        "ENTER: compile + run",
        "SPACE: pause/resume",
        "C: add clap at source",
        "R: back to setup",
        "ESC: quit",
        "",
        f"status: {status}",
    ]

    y = panel_y + 12
    prev_clip = screen.get_clip()
    screen.set_clip(pygame.Rect(text_left, panel_y + 8, max(10, text_right - text_left), panel_h - 16))
    for idx, text in enumerate(lines):
        use_small = idx >= 2
        f = small_font if use_small else font
        color = TEXT_COLOR
        if text.startswith("1)") and selected_index == 0:
            color = HIGHLIGHT_COLOR
        if text.startswith("2)") and selected_index == 1:
            color = HIGHLIGHT_COLOR
        if text.startswith("3)") and selected_index == 2:
            color = HIGHLIGHT_COLOR
        if text.startswith("4)") and selected_index == 3:
            color = HIGHLIGHT_COLOR
        if text.startswith("5)") and selected_index == 4:
            color = HIGHLIGHT_COLOR
        if text.startswith("6)") and selected_index == 5:
            color = HIGHLIGHT_COLOR
        wrapped = wrap_text(f, text, text_right - text_left)
        for wrapped_line in wrapped:
            surf = f.render(wrapped_line, True, color)
            screen.blit(surf, (text_left, y))
            y += surf.get_height() + 2
        y += 1
    screen.set_clip(prev_clip)


def apply_param_delta(params: SimParams, index: int, direction: int) -> None:
    if index == 0:
        params.amp = clamp_float(params.amp + 0.1 * direction, 0.1, 5.0)
    elif index == 1:
        params.freq_hz = clamp_float(params.freq_hz + 10.0 * direction, 50.0, 2000.0)
    elif index == 2:
        params.beta = clamp_float(params.beta + 0.01 * direction, 0.0, 5.0)
    elif index == 3:
        params.steps_per_frame = clamp_int(params.steps_per_frame + direction, 1, 30)
    elif index == 4:
        params.fps = clamp_int(params.fps + direction, 20, 144)
    elif index == 5:
        params.reclap_every = clamp_int(params.reclap_every + 50 * direction, 100, 5000)


def draw_source_marker(screen: pygame.Surface, row: int | None, col: int | None) -> None:
    if row is None or col is None:
        return
    px, py = cell_to_screen_center(row, col)
    pygame.draw.circle(screen, SOURCE_COLOR, (px, py), max(4, FLOOR_PLAN_CELL_SIZE // 2), width=2)
    pygame.draw.line(screen, SOURCE_COLOR, (px - 8, py), (px + 8, py), 2)
    pygame.draw.line(screen, SOURCE_COLOR, (px, py - 8), (px, py + 8), 2)


def main(smoke: bool = False) -> None:
    if smoke:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    pygame.init()
    window_w = FLOOR_PLAN_SCREEN_W + EXTRA_RIGHT_PANEL_PX
    screen = pygame.display.set_mode((window_w, FLOOR_PLAN_SCREEN_H))
    pygame.display.set_caption("Acoustic Simulator - floor-plan driven")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 26)
    small_font = pygame.font.SysFont(None, 20)

    alpha = create_floor_plan().astype(np.float64)
    alpha_surface = pygame.transform.scale(
        to_surface(alpha_to_rgb(alpha)),
        (FLOOR_PLAN_GRID_PX_W, FLOOR_PLAN_GRID_PX_H),
    )

    params = SimParams()
    source_row: int | None = None
    source_col: int | None = None
    selected_param = 0
    status = "Click in the grid to place the source, then press Enter."

    mode = "setup"
    session: CompiledSession | None = None
    running = True
    frames = 0

    if smoke:
        source_row = FLOOR_PLAN_NY // 2
        source_col = FLOOR_PLAN_NX // 2
        if alpha[source_row, source_col] > 0.0:
            air_cells = np.argwhere(alpha == 0.0)
            source_row, source_col = map(int, air_cells[len(air_cells) // 2])
        session = compile_session(params, source_row, source_col)
        mode = "running"
        status = "Smoke mode: auto-compiled and running."

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_UP:
                    selected_param = (selected_param - 1) % 6
                elif ev.key == pygame.K_DOWN:
                    selected_param = (selected_param + 1) % 6
                elif ev.key == pygame.K_LEFT:
                    apply_param_delta(params, selected_param, -1)
                    if mode != "setup":
                        status = "Parameter changed. Press R then Enter to rebuild."
                elif ev.key == pygame.K_RIGHT:
                    apply_param_delta(params, selected_param, 1)
                    if mode != "setup":
                        status = "Parameter changed. Press R then Enter to rebuild."
                elif ev.key == pygame.K_RETURN:
                    if source_row is None or source_col is None:
                        status = "Choose a source cell first."
                    elif alpha[source_row, source_col] > 0.0:
                        status = "Source must be in air, not on a wall."
                    else:
                        try:
                            session = compile_session(params, source_row, source_col)
                            mode = "running"
                            status = "Simulation compiled from floor plan and running."
                        except Exception as exc:
                            status = f"Compile failed: {exc}"
                elif ev.key == pygame.K_SPACE and mode in {"running", "paused"}:
                    mode = "paused" if mode == "running" else "running"
                    status = "Paused." if mode == "paused" else "Resumed."
                elif ev.key == pygame.K_r:
                    mode = "setup"
                    session = None
                    status = "Back in setup mode. Press Enter to compile."
                elif ev.key == pygame.K_c and session is not None:
                    sx = session.source_col * session.grid.dx
                    sy = session.source_row * session.grid.dy
                    session.solver.add_impulse(sx, sy, amp=params.amp, freq=params.freq_hz)
                    status = "Injected a clap at the selected source."

            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                rc = screen_to_cell(*ev.pos)
                if rc is not None:
                    row, col = rc
                    source_row, source_col = row, col
                    if alpha[row, col] > 0.0:
                        status = "Selected a wall cell. Pick an air cell for source."
                    else:
                        status = f"Source set to cell ({row}, {col})."

        screen.fill((24, 24, 24))
        screen.blit(alpha_surface, (FLOOR_PLAN_MARGIN_LEFT, FLOOR_PLAN_MARGIN_TOP))

        if session is not None:
            if mode == "running":
                for _ in range(params.steps_per_frame):
                    session.solver.step()
                if session.solver.n % params.reclap_every < params.steps_per_frame:
                    sx = session.source_col * session.grid.dx
                    sy = session.source_row * session.grid.dy
                    session.solver.add_impulse(sx, sy, amp=params.amp, freq=params.freq_hz)

            field = session.solver.field
            peak = float(np.max(np.abs(field)))
            session.p_scale = max(peak, session.p_scale * 0.97, 1e-4)

            sim_surface = pygame.transform.scale(
                to_surface(pressure_to_rgb(field, session.alpha, session.p_scale)),
                (FLOOR_PLAN_GRID_PX_W, FLOOR_PLAN_GRID_PX_H),
            )
            screen.blit(sim_surface, (FLOOR_PLAN_MARGIN_LEFT, FLOOR_PLAN_MARGIN_TOP))

        draw_source_marker(screen, source_row, source_col)

        steps = 0 if session is None else session.solver.n
        sim_time = 0.0 if session is None else session.solver.t * 1e3
        legend_scale = 1e-4 if session is None else session.p_scale
        draw_panel(
            screen,
            font,
            small_font,
            params,
            source_row,
            source_col,
            status,
            selected_param,
            mode,
            steps,
            sim_time,
            alpha,
            legend_scale,
        )

        pygame.display.flip()
        clock.tick(params.fps)

        frames += 1
        if smoke and frames >= 120:
            running = False

    pygame.quit()

    if smoke:
        if session is None:
            print("smoke OK: setup mode rendered with floor plan")
        else:
            f = session.solver.field
            print(
                f"smoke OK: {frames} frames, {session.solver.n} steps, "
                f"max|u|={float(np.max(np.abs(f))):.4f}, "
                f"finite={bool(np.all(np.isfinite(f)))}"
            )


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
