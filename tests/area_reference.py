"""Independent exact-area reference used ONLY by the test-suite.

Nothing in this module imports :mod:`winding_fill`.  Expected values are
computed by a different code path from the package rasterizer:

* a *global* vertical-slab decomposition over the whole query window
  (no per-pixel Liang-Barsky clipping -- instead, slab events include
  every edge endpoint, every edge/window-boundary crossing and every
  pairwise edge crossing);
* winding numbers sampled with an *upward vertical* ray and a half-open
  x rule (the package uses a horizontal ray with a half-open y rule);
* all arithmetic in :class:`fractions.Fraction`, so results are exact.

The covered set ``{winding != 0}`` is convention-independent, so both
implementations must agree exactly if both are correct.
"""

from __future__ import annotations

from fractions import Fraction

_ZERO = Fraction(0)
_ONE = Fraction(1)


def _edges_from_raw(raw_contours):
    """Parse raw line-segment contours into Fraction edges."""
    edges = []
    for raw_con in raw_contours:
        for seg in raw_con:
            if seg["type"] != "line":
                raise ValueError("reference handles line segments only")
            edges.append(
                (
                    Fraction(seg["p0"]["x"]),
                    Fraction(seg["p0"]["y"]),
                    Fraction(seg["p1"]["x"]),
                    Fraction(seg["p1"]["y"]),
                )
            )
    return edges


def _winding_up(edges, sx, sy):
    """Signed winding via an upward vertical ray, half-open x rule."""
    winding = 0
    for x0, y0, x1, y1 in edges:
        if x0 == x1:
            continue
        lo, hi = (x0, x1) if x0 < x1 else (x1, x0)
        if not (lo <= sx < hi):
            continue
        yc = y0 + (sx - x0) * (y1 - y0) / (x1 - x0)
        if yc >= sy:
            winding += 1 if x1 > x0 else -1
    return winding


def _cross_x(e1, e2):
    ax, ay, bx, by = e1
    cx, cy, dx, dy = e2
    d1x, d1y = bx - ax, by - ay
    d2x, d2y = dx - cx, dy - cy
    den = d1x * d2y - d1y * d2x
    if den == 0:
        return None
    t = ((cx - ax) * d2y - (cy - ay) * d2x) / den
    s = ((cx - ax) * d1y - (cy - ay) * d1x) / den
    if _ZERO <= t <= _ONE and _ZERO <= s <= _ONE:
        return ax + t * d1x
    return None


def reference_window_area(raw_contours, ox, oy, width, height):
    """Exact area of the nonzero-winding region clipped to the window."""
    edges = _edges_from_raw(raw_contours)
    wx0, wy0 = Fraction(ox), Fraction(oy)
    wx1, wy1 = wx0 + width, wy0 + height

    xs = {wx0, wx1}
    for x0, y0, x1, y1 in edges:
        for x in (x0, x1):
            if wx0 < x < wx1:
                xs.add(x)
        if y0 != y1:
            lo, hi = (y0, y1) if y0 < y1 else (y1, y0)
            for wy in (wy0, wy1):
                if lo < wy < hi:
                    xc = x0 + (wy - y0) * (x1 - x0) / (y1 - y0)
                    if wx0 < xc < wx1:
                        xs.add(xc)
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            x = _cross_x(edges[i], edges[j])
            if x is not None and wx0 < x < wx1:
                xs.add(x)
    xs = sorted(xs)

    total = _ZERO
    for xa, xb in zip(xs, xs[1:]):
        mid = (xa + xb) / 2
        bounds = []
        for x0, y0, x1, y1 in edges:
            if x0 == x1:
                continue
            lo_x, hi_x = (x0, x1) if x0 < x1 else (x1, x0)
            if lo_x <= xa and hi_x >= xb:
                ya = y0 + (xa - x0) * (y1 - y0) / (x1 - x0)
                yb = y0 + (xb - x0) * (y1 - y0) / (x1 - x0)
                if ya <= wy0 and yb <= wy0:
                    continue  # fully below the window in this slab
                if ya >= wy1 and yb >= wy1:
                    continue  # fully above the window in this slab
                bounds.append(((ya + yb) / 2, ya, yb))
        bounds.sort()
        bounds = [(wy0, wy0, wy0)] + bounds + [(wy1, wy1, wy1)]
        for lo, hi in zip(bounds, bounds[1:]):
            if hi[0] <= lo[0]:
                continue
            sy = (lo[0] + hi[0]) / 2
            if _winding_up(edges, mid, sy) != 0:
                total += (xb - xa) * ((hi[1] - lo[1]) + (hi[2] - lo[2])) / 2
    return total


def reference_pixel_area(raw_contours, px, py):
    """Exact nonzero-winding area inside one pixel's unit square."""
    return reference_window_area(raw_contours, px, py, 1, 1)
