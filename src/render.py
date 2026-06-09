"""
render.py — shared field -> pixels mapping for the Pygame views (visual layer).

Monochrome diverging theme (white / black): air rests at neutral GRAY, compression
goes toward WHITE, rarefaction toward BLACK; obstacles are near-black with a light
outline so they read as structure.  Imported by both `visualiser.py` (the interactive
app) and `pygame_demo.py` (the auto demo) so the two views look identical.
"""

import numpy as np
import pygame

# Canvas palette (grayscale)
AIR = 128            # rest pressure  -> mid gray
WALL = 22            # obstacle fill  -> near black
WALL_EDGE = 120      # obstacle outline -> mid gray (a crisp "structure" edge)


def field_to_rgb(field, alpha, p_scale, wall_rgb=None, gamma=0.6):
    """(NY, NX) pressure + (NY, NX) alpha-map  ->  (NY, NX, 3) uint8 image.

    Wave:  rest/neutral pressure is WHITE; compression (u > 0) reddens toward pure red
    to mark the wavefront; rarefaction (u < 0) stays white (it blends into the
    background).  A gamma < 1 lift keeps faint fronts and echoes visible:
        pos = clip(u/p_scale, 0, 1) ** gamma ;   R = 255 ,  G = B = 255·(1 − pos)
    Obstacles (alpha > 0): coloured per `wall_rgb` (NY,NX,3) if supplied, otherwise
    near-black grayscale with a light outline.
    """
    pos = np.clip(field / p_scale, 0.0, 1.0) ** gamma    # compression only
    chan = (255.0 * (1.0 - pos)).astype(np.uint8)
    rgb = np.empty(field.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = 255                                    # red channel always full
    rgb[..., 1] = chan                                   # green/blue fade as compression rises
    rgb[..., 2] = chan

    solid = alpha > 0.0
    if wall_rgb is not None:
        rgb[solid] = wall_rgb[solid]                     # colour walls by material
    else:
        rgb[solid] = (WALL, WALL, WALL)
        air = ~solid
        pa = np.pad(air, 1, mode="constant", constant_values=False)
        air_adj = pa[2:, 1:-1] | pa[:-2, 1:-1] | pa[1:-1, 2:] | pa[1:-1, :-2]
        rgb[solid & air_adj] = (WALL_EDGE, WALL_EDGE, WALL_EDGE)
    return rgb


def to_surface(rgb):
    """(NY, NX, 3) -> pygame Surface with y pointing up (grid row 0 at the window bottom)."""
    flipped = rgb[::-1]                                          # y=0 at the bottom
    arr = np.ascontiguousarray(np.transpose(flipped, (1, 0, 2)))  # -> (NX, NY, 3)
    return pygame.surfarray.make_surface(arr)


def draw_line_chart(surface, rect, values, *, times=None, font=None,
                    color=(28, 28, 28), bg=(255, 255, 255), axis=(200, 200, 200),
                    label=(120, 120, 140), y_max=None):
    """Draw a polyline chart of `values` inside `rect` (white panel, gray frame).

    When `font` is given, numeric tick labels are drawn on both axes: the y-axis is
    labelled 0 / vmax/2 / vmax (the energy scale), and the x-axis is labelled in
    milliseconds spanning `times[0]`..`times[-1]` (the elapsed simulation time) — so the
    chart reports real numbers, not just a shape.  `times` must run parallel to `values`.

    Returns the y-axis maximum used (so callers can label it).  Drawn with pygame
    primitives only — no matplotlib.
    """
    x, y, w, h = rect
    pygame.draw.rect(surface, bg, rect)
    pygame.draw.rect(surface, axis, rect, 1)

    # Reserve margins for tick labels only when we actually have a font to draw them.
    ml, mb, mt, mr = (44, 16, 6, 10) if font is not None else (0, 0, 0, 0)
    px0, py0 = x + ml, y + mt                       # plot-area top-left
    pw, ph = w - ml - mr, h - mt - mb               # plot-area size

    n = len(values)
    vmax = y_max if y_max is not None else (max(values) if n else 1.0)
    vmax = vmax if vmax and vmax > 0 else 1.0

    # L-shaped axes around the inner plot area.
    pygame.draw.line(surface, axis, (px0, py0), (px0, py0 + ph))
    pygame.draw.line(surface, axis, (px0, py0 + ph), (px0 + pw, py0 + ph))

    if font is not None:
        for frac in (0.0, 0.5, 1.0):                # y ticks: 0, half, full energy
            ty = py0 + ph * (1.0 - frac)
            pygame.draw.line(surface, axis, (px0 - 3, ty), (px0, ty))
            lab = font.render(f"{vmax * frac:.3g}", True, label)
            surface.blit(lab, (x, ty - lab.get_height() / 2))
        if times and len(times) >= 2:               # x ticks: time across the run, in ms
            t0, t1 = times[0], times[-1]
            for frac in (0.0, 1.0 / 3, 2.0 / 3, 1.0):
                tx = px0 + pw * frac
                pygame.draw.line(surface, axis, (tx, py0 + ph), (tx, py0 + ph + 3))
                tms = (t0 + (t1 - t0) * frac) * 1e3
                lab = font.render(f"{tms:.0f} ms" if frac == 1.0 else f"{tms:.0f}", True, label)
                lr = lab.get_rect(midtop=(tx, py0 + ph + 2))
                lr.right = min(lr.right, x + w)      # keep the last label inside the panel
                surface.blit(lab, lr)

    if n < 2:
        return vmax

    # Downsample so the polyline never has more vertices than the plot has pixels.
    step = max(1, n // int(pw)) if pw > 0 else 1
    pts = []
    for i in range(0, n, step):
        gx = px0 + (pw - 1) * i / (n - 1)
        gy = py0 + (ph - 1) * (1.0 - min(max(values[i], 0.0) / vmax, 1.0))
        pts.append((gx, gy))
    if (n - 1) % step != 0:                          # always include the latest sample
        gy = py0 + (ph - 1) * (1.0 - min(max(values[-1], 0.0) / vmax, 1.0))
        pts.append((px0 + (pw - 1), gy))
    pygame.draw.lines(surface, color, False, pts, 2)
    return vmax
