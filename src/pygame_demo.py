"""
pygame_demo.py — visual test harness for the acoustic FDTD physics.

Renders the constructed room (the alpha-map: walls / furniture / doorways) and
animates the pressure field as the solver steps.  It auto-claps and loops so you
can watch wavefronts expand, reflect off walls, pass through the doorway, and die
out faster against absorptive materials.

This is a *physics* test view.  USER INTERACTION — click-to-place source, material
picker, sliders, receivers — is deliberately NOT implemented here; that belongs to
the teammate's `visualiser.py`.  The only events handled are window-close / ESC.
(The teammate hooks click-to-place in where marked below: pixel -> cell -> metres
-> `solver.add_impulse(x, y)`.)

Run:
    cd src
    python pygame_demo.py
    python pygame_demo.py --smoke      # headless self-check, no window
"""

import os
import sys

import numpy as np
import pygame

from grid import RoomGrid
from room import Room
from physics_solver import WaveSolver

SCALE = 6                 # screen pixels per grid cell
STEPS_PER_FRAME = 3       # FDTD steps advanced per rendered frame (slow-mo to watch)
RECLAP_EVERY = 1200       # re-trigger a clap every N steps so the demo loops
FPS = 60


# ── scene ─────────────────────────────────────────────────────────────────────
def build_room(grid: RoomGrid) -> Room:
    """Two rooms in a concrete shell, a wooden partition with a doorway, a sofa."""
    return (Room(grid)
            .add_border("Concrete")
            .add_rectangle("Wood", 4.0, 0.0, 4.15, 6.0)   # partition wall at x ~ 4 m
            .add_doorway(4.0, 2.5, 4.15, 3.5)             # 1 m doorway gap
            .add_block("Carpet", 1.0, 1.0, 2.5, 2.0))     # soft furniture (sofa)


# ── rendering ─────────────────────────────────────────────────────────────────
def render_rgb(field, alpha, p_scale):
    """Compose an (NY, NX, 3) uint8 image: pressure on black, obstacles in grey.

    Compression (+) is red, rarefaction (-) is blue, still air is near-black so the
    wavefronts stand out; walls/furniture are grey (lighter = more reflective).
    """
    val = field / p_scale
    pos = np.clip(val, 0.0, 1.0)
    neg = np.clip(-val, 0.0, 1.0)

    rgb = np.empty(field.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = (20 + 235 * pos).astype(np.uint8)   # red  for compression
    rgb[..., 1] = 20
    rgb[..., 2] = (20 + 235 * neg).astype(np.uint8)   # blue for rarefaction

    solid = alpha > 0.0
    shade = (30 + 220 * (1.0 - alpha)).astype(np.uint8)   # rigid=light, absorptive=dark
    sh = shade[solid]
    rgb[solid] = np.stack([sh, sh, sh], axis=1)
    return rgb


def to_surface(rgb):
    """(NY, NX, 3) -> pygame Surface, y pointing up (grid row 0 at the window bottom)."""
    flipped = rgb[::-1]                                # put y=0 at the bottom
    arr = np.ascontiguousarray(np.transpose(flipped, (1, 0, 2)))   # -> (NX, NY, 3)
    return pygame.surfarray.make_surface(arr)


# ── main loop ─────────────────────────────────────────────────────────────────
def main(smoke=False):
    if smoke:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    grid = RoomGrid()
    room = build_room(grid)
    solver = WaveSolver(grid, room.alpha)

    clap_points = [(2.0, 3.0), (6.0, 3.0)]            # alternate left / right room
    clap_idx = [0]

    def clap():
        x, y = clap_points[clap_idx[0] % len(clap_points)]
        solver.add_impulse(x, y, amp=1.0)
        clap_idx[0] += 1

    clap()                                            # first clap at t = 0

    pygame.init()
    win = (grid.NX * SCALE, grid.NY * SCALE)
    screen = pygame.display.set_mode(win)
    pygame.display.set_caption("Acoustic FDTD - physics test view  (close / ESC to quit)")
    clock = pygame.time.Clock()

    p_scale = 1e-4
    running = True
    frames = 0
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False
            # ---- teammate's visualiser.py adds interaction here, e.g.:
            #   elif ev.type == pygame.MOUSEBUTTONDOWN:
            #       px, py = ev.pos
            #       col = px // SCALE
            #       row = (grid.NY - 1) - (py // SCALE)        # undo the y-flip
            #       solver.add_impulse(col * grid.dx, row * grid.dy)

        for _ in range(STEPS_PER_FRAME):
            solver.step()
        if solver.n % RECLAP_EVERY < STEPS_PER_FRAME:
            clap()

        field = solver.field
        peak = float(np.max(np.abs(field)))
        p_scale = max(peak, p_scale * 0.97, 1e-4)     # adaptive, lightly smoothed

        surf = to_surface(render_rgb(field, room.alpha, p_scale))
        screen.blit(pygame.transform.scale(surf, win), (0, 0))
        pygame.display.flip()
        clock.tick(FPS)

        frames += 1
        if smoke and frames >= 60:
            running = False

    pygame.quit()
    if smoke:
        f = solver.field
        print(f"smoke OK: {frames} frames, {solver.n} steps, "
              f"max|u|={float(np.max(np.abs(f))):.4f}, "
              f"finite={bool(np.all(np.isfinite(f)))}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
