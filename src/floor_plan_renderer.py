"""Pygame renderer for the floor-plan absorption grid."""

import numpy as np
import pygame

from config import (
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
)


_YLRD_ANCHORS = [
    (0.000, (255, 255, 204)),
    (0.125, (255, 237, 160)),
    (0.250, (254, 217, 118)),
    (0.375, (254, 178, 76)),
    (0.500, (253, 141, 60)),
    (0.625, (252, 78, 42)),
    (0.750, (227, 26, 28)),
    (0.875, (189, 0, 38)),
    (1.000, (128, 0, 38)),
]

_PRESSURE_HEATMAP_ANCHORS = [
    (0.000, (33, 102, 172)),
    (0.250, (67, 147, 195)),
    (0.500, (146, 197, 222)),
    (0.750, (253, 219, 199)),
    (1.000, (178, 24, 43)),
]


def _interpolate_color(anchors: list[tuple[float, tuple[int, int, int]]], t: float) -> tuple:
    t = max(0.0, min(1.0, t))
    for i in range(len(anchors) - 1):
        t0, c0 = anchors[i]
        t1, c1 = anchors[i + 1]
        if t <= t1:
            s = (t - t0) / (t1 - t0)
            return (
                int(c0[0] + s * (c1[0] - c0[0])),
                int(c0[1] + s * (c1[1] - c0[1])),
                int(c0[2] + s * (c1[2] - c0[2])),
            )
    return anchors[-1][1]

def _ylrd_color(t: float) -> tuple:
    return _interpolate_color(_YLRD_ANCHORS, t)


def _pressure_color(t: float) -> tuple:
    return _interpolate_color(_PRESSURE_HEATMAP_ANCHORS, t)


def _alpha_to_color(alpha: float, vmax: float) -> tuple:
    if alpha == 0.0:
        return (255, 255, 255)
    return _ylrd_color(alpha / vmax)

def generate_heatmap_image(
    values: np.ndarray,
    title: str = "Pressure Heatmap",
    cell_size: int = 8,
    save_path: str | None = None,
) -> pygame.Surface:
    """
    Render a 2D scalar field as a heatmap image.

    Args:
        values: 2D array where each element is the scalar value for one cell.
        title: Text drawn above the heatmap.
        cell_size: Pixel size for each array element.
        save_path: Optional file path to save the rendered image.

    Returns:
        A pygame.Surface containing the rendered heatmap.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"values must be a 2D array, got shape {values.shape!r}")

    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()

    rows, cols = values.shape
    heatmap_w = cols * cell_size
    heatmap_h = rows * cell_size

    margin_left = 24
    margin_right = 118
    margin_top = 52
    margin_bottom = 36

    screen_w = margin_left + heatmap_w + margin_right
    screen_h = margin_top + heatmap_h + margin_bottom

    scene = pygame.Surface((screen_w, screen_h))
    scene.fill((236, 236, 236))

    font_sm = pygame.font.SysFont(None, 18)
    font_md = pygame.font.SysFont(None, 20)
    font_lg = pygame.font.SysFont(None, 26, bold=True)

    value_min = float(np.min(values))
    value_max = float(np.max(values))
    value_span = value_max - value_min

    pygame.draw.rect(scene, (255, 255, 255), (margin_left, margin_top, heatmap_w, heatmap_h))

    for row in range(rows):
        screen_py = margin_top + (rows - 1 - row) * cell_size
        for col in range(cols):
            if value_span == 0.0:
                t = 0.0
            else:
                t = (values[row, col] - value_min) / value_span
            color = _pressure_color(t)
            screen_px = margin_left + col * cell_size
            pygame.draw.rect(scene, color, (screen_px, screen_py, cell_size, cell_size))

    pygame.draw.rect(scene, (0, 0, 0), (margin_left, margin_top, heatmap_w, heatmap_h), 1)

    if cell_size >= 4:
        grid_color = (255, 255, 255, 40)
        for col in range(1, cols):
            px = margin_left + col * cell_size
            pygame.draw.line(scene, grid_color, (px, margin_top), (px, margin_top + heatmap_h))
        for row in range(1, rows):
            py = margin_top + row * cell_size
            pygame.draw.line(scene, grid_color, (margin_left, py), (margin_left + heatmap_w, py))

    title_surf = font_lg.render(title, True, (0, 0, 0))
    scene.blit(
        title_surf,
        (
            margin_left + heatmap_w // 2 - title_surf.get_width() // 2,
            margin_top // 2 - title_surf.get_height() // 2,
        ),
    )

    cbar_x = margin_left + heatmap_w + 22
    cbar_y = margin_top
    cbar_w = 18
    cbar_h = heatmap_h

    for i in range(cbar_h):
        t = 1.0 - i / max(1, cbar_h - 1)
        color = _pressure_color(t)
        pygame.draw.line(scene, color, (cbar_x, cbar_y + i), (cbar_x + cbar_w - 1, cbar_y + i))
    pygame.draw.rect(scene, (0, 0, 0), (cbar_x, cbar_y, cbar_w, cbar_h), 1)

    for val in np.linspace(value_min, value_max, 6):
        if value_span == 0.0:
            ty = cbar_y + cbar_h // 2
        else:
            ty = cbar_y + int((1.0 - (val - value_min) / value_span) * cbar_h)
        ty = max(cbar_y, min(cbar_y + cbar_h, ty))
        pygame.draw.line(scene, (0, 0, 0), (cbar_x + cbar_w, ty), (cbar_x + cbar_w + 4, ty))
        lbl = font_sm.render(f"{val:.3f}", True, (0, 0, 0))
        scene.blit(lbl, (cbar_x + cbar_w + 6, ty - lbl.get_height() // 2))

    cl = font_md.render("pressure", True, (0, 0, 0))
    cl_rot = pygame.transform.rotate(cl, -90)
    scene.blit(
        cl_rot,
        (
            screen_w - cl_rot.get_width() - 2,
            margin_top + heatmap_h // 2 - cl_rot.get_height() // 2,
        ),
    )

    if save_path is not None:
        pygame.image.save(scene, save_path)

    return scene

def _build_scene(grid: np.ndarray, title: str) -> pygame.Surface:
    scene = pygame.Surface((FLOOR_PLAN_SCREEN_W, FLOOR_PLAN_SCREEN_H))
    scene.fill((210, 210, 210))

    font_sm = pygame.font.SysFont(None, 18)
    font_md = pygame.font.SysFont(None, 20)
    font_lg = pygame.font.SysFont(None, 26, bold=True)

    vmax = max(0.15, float(np.max(grid)))

    pygame.draw.rect(
        scene,
        (255, 255, 255),
        (FLOOR_PLAN_MARGIN_LEFT, FLOOR_PLAN_MARGIN_TOP, FLOOR_PLAN_GRID_PX_W, FLOOR_PLAN_GRID_PX_H),
    )

    for row in range(FLOOR_PLAN_NY):
        row_data = grid[row]
        screen_py = FLOOR_PLAN_MARGIN_TOP + (FLOOR_PLAN_NY - 1 - row) * FLOOR_PLAN_CELL_SIZE
        for col in range(FLOOR_PLAN_NX):
            alpha = row_data[col]
            if alpha == 0.0:
                continue
            color = _alpha_to_color(alpha, vmax)
            screen_px = FLOOR_PLAN_MARGIN_LEFT + col * FLOOR_PLAN_CELL_SIZE
            pygame.draw.rect(scene, color, (screen_px, screen_py, FLOOR_PLAN_CELL_SIZE, FLOOR_PLAN_CELL_SIZE))

    pygame.draw.rect(
        scene,
        (0, 0, 0),
        (FLOOR_PLAN_MARGIN_LEFT, FLOOR_PLAN_MARGIN_TOP, FLOOR_PLAN_GRID_PX_W, FLOOR_PLAN_GRID_PX_H),
        1,
    )

    t_surf = font_lg.render(title, True, (0, 0, 0))
    scene.blit(
        t_surf,
        (
            FLOOR_PLAN_MARGIN_LEFT + FLOOR_PLAN_GRID_PX_W // 2 - t_surf.get_width() // 2,
            FLOOR_PLAN_MARGIN_TOP // 2 - t_surf.get_height() // 2,
        ),
    )

    axis_y = FLOOR_PLAN_MARGIN_TOP + FLOOR_PLAN_GRID_PX_H
    for x_m in np.arange(0.0, FLOOR_PLAN_ROOM_WIDTH_M + 1e-9, 1.0):
        px = FLOOR_PLAN_MARGIN_LEFT + int(round(x_m / FLOOR_PLAN_DX)) * FLOOR_PLAN_CELL_SIZE
        pygame.draw.line(scene, (0, 0, 0), (px, axis_y), (px, axis_y + 5))
        lbl = font_sm.render(f"{x_m:.0f}", True, (0, 0, 0))
        scene.blit(lbl, (px - lbl.get_width() // 2, axis_y + 8))
    xl = font_md.render("x [m]", True, (0, 0, 0))
    scene.blit(xl, (FLOOR_PLAN_MARGIN_LEFT + FLOOR_PLAN_GRID_PX_W // 2 - xl.get_width() // 2, axis_y + 26))

    axis_x = FLOOR_PLAN_MARGIN_LEFT
    for y_m in np.arange(0.0, FLOOR_PLAN_ROOM_HEIGHT_M + 1e-9, 1.0):
        py = FLOOR_PLAN_MARGIN_TOP + FLOOR_PLAN_GRID_PX_H - int(round(y_m / FLOOR_PLAN_DX)) * FLOOR_PLAN_CELL_SIZE
        pygame.draw.line(scene, (0, 0, 0), (axis_x - 5, py), (axis_x, py))
        lbl = font_sm.render(f"{y_m:.0f}", True, (0, 0, 0))
        scene.blit(lbl, (axis_x - lbl.get_width() - 8, py - lbl.get_height() // 2))
    yl = font_md.render("y [m]", True, (0, 0, 0))
    yl_rot = pygame.transform.rotate(yl, 90)
    scene.blit(yl_rot, (6, FLOOR_PLAN_MARGIN_TOP + FLOOR_PLAN_GRID_PX_H // 2 - yl_rot.get_height() // 2))

    cbar_x = FLOOR_PLAN_MARGIN_LEFT + FLOOR_PLAN_GRID_PX_W + 22
    cbar_y = FLOOR_PLAN_MARGIN_TOP
    cbar_w = 18
    cbar_h = FLOOR_PLAN_GRID_PX_H

    for i in range(cbar_h):
        t = 1.0 - i / cbar_h
        color = _ylrd_color(t)
        pygame.draw.line(scene, color, (cbar_x, cbar_y + i), (cbar_x + cbar_w - 1, cbar_y + i))
    pygame.draw.rect(scene, (0, 0, 0), (cbar_x, cbar_y, cbar_w, cbar_h), 1)

    for val in np.linspace(0.0, vmax, 6):
        ty = cbar_y + int((1.0 - val / vmax) * cbar_h)
        ty = max(cbar_y, min(cbar_y + cbar_h, ty))
        pygame.draw.line(scene, (0, 0, 0), (cbar_x + cbar_w, ty), (cbar_x + cbar_w + 4, ty))
        lbl = font_sm.render(f"{val:.3f}", True, (0, 0, 0))
        scene.blit(lbl, (cbar_x + cbar_w + 6, ty - lbl.get_height() // 2))

    cl = font_md.render("alpha (absorption)", True, (0, 0, 0))
    cl_rot = pygame.transform.rotate(cl, -90)
    scene.blit(
        cl_rot,
        (
            FLOOR_PLAN_SCREEN_W - cl_rot.get_width() - 2,
            FLOOR_PLAN_MARGIN_TOP + FLOOR_PLAN_GRID_PX_H // 2 - cl_rot.get_height() // 2,
        ),
    )

    return scene


def show_grid(grid: np.ndarray, title: str = "Floor Plan") -> None:
    pygame.init()
    screen = pygame.display.set_mode((FLOOR_PLAN_SCREEN_W, FLOOR_PLAN_SCREEN_H))
    pygame.display.set_caption(title)

    scene = _build_scene(grid, title)

    clock = pygame.time.Clock()
    running = True
    while running:
        screen.blit(scene, (0, 0))
        pygame.display.flip()
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()
