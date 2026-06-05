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


def _ylrd_color(t: float) -> tuple:
    t = max(0.0, min(1.0, t))
    for i in range(len(_YLRD_ANCHORS) - 1):
        t0, c0 = _YLRD_ANCHORS[i]
        t1, c1 = _YLRD_ANCHORS[i + 1]
        if t <= t1:
            s = (t - t0) / (t1 - t0)
            return (
                int(c0[0] + s * (c1[0] - c0[0])),
                int(c0[1] + s * (c1[1] - c0[1])),
                int(c0[2] + s * (c1[2] - c0[2])),
            )
    return _YLRD_ANCHORS[-1][1]


def _alpha_to_color(alpha: float, vmax: float) -> tuple:
    if alpha == 0.0:
        return (255, 255, 255)
    return _ylrd_color(alpha / vmax)


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
