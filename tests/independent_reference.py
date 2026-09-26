"""Independent reference implementations used ONLY by the test-suite.

Nothing in this module imports :mod:`winding_fill`.  Expected values are
computed by a completely different code path from the rasterizer core:

* :func:`reference_levels_lines` -- exact rational arithmetic
  (:class:`fractions.Fraction`), row-by-row ray casting.  Every crossing
  x and every comparison is exact, so for straight-line geometry the
  expected coverage/winding values are mathematically exact, not "close".

* :func:`reference_levels_mixed` -- high-precision decimal arithmetic
  (:class:`decimal.Decimal`, 60 digits) for contours that may contain
  quadratic Beziers.  Instead of the core's closed-form quadratic
  formula, roots of y(t) = sy are found by bisection on each
  y-monotonic piece (de Casteljau evaluation), an independent numerical
  method.

Both references implement the same contractual rules as the core:
half-open y interval ``lo <= sy < hi``, signed upward/downward
crossings, horizontal pieces ignored, nonzero winding => covered.
"""

from __future__ import annotations

import bisect
from decimal import Decimal, getcontext
from fractions import Fraction
from typing import List, Tuple

getcontext().prec = 60

_SAMPLE_OFFSETS = [Fraction(2 * i + 1, 8) for i in range(4)]
_SAMPLE_OFFSETS_D = [Decimal(2 * i + 1) / Decimal(8) for i in range(4)]
_BISECTION_STEPS = 130  # 2^-130 ~ 1e-39, vastly finer than float precision


# ---------------------------------------------------------------------------
# Shared half-open-y crossing rule
# ---------------------------------------------------------------------------


def _span_and_dir(ya, yb):
    """Return (lo, hi, direction) for the half-open rule, or None."""
    if yb > ya:
        return ya, yb, 1
    if yb < ya:
        return yb, ya, -1
    return None


def _assemble(rows_by_py, width, height, decimal=False):
    """Turn {py: {j: [(x, dir), ...]}} into (levels, windings)."""
    zero = _D(0) if decimal else Fraction(0)
    offsets = _SAMPLE_OFFSETS_D if decimal else _SAMPLE_OFFSETS
    sample_x = (lambda px, i: zero + Decimal(px) + offsets[i]) if decimal else (
        lambda px, i: zero + Fraction(px) + offsets[i]
    )
    levels_rows: List[Tuple[int, ...]] = []
    wind_rows: List[Tuple[Tuple[int, ...], ...]] = []
    for py in range(height):
        by_px: List[List[int]] = [[0] * 16 for _ in range(width)]
        for j in range(4):
            crossings = sorted(rows_by_py[py][j], key=lambda h: h[0])
            xs = [c[0] for c in crossings]
            dirs = [c[1] for c in crossings]
            suffix = [0] * (len(dirs) + 1)
            for k in range(len(dirs) - 1, -1, -1):
                suffix[k] = suffix[k + 1] + dirs[k]
            for px in range(width):
                for i in range(4):
                    sx = sample_x(px, i)
                    k = bisect.bisect_left(xs, sx)
                    by_px[px][j * 4 + i] = suffix[k]
        levels_rows.append(
            tuple(sum(1 for w in by_px[px] if w != 0) for px in range(width))
        )
        wind_rows.append(tuple(tuple(by_px[px]) for px in range(width)))
    return tuple(levels_rows), tuple(wind_rows)


# ---------------------------------------------------------------------------
# Exact reference: line segments only, rational arithmetic
# ---------------------------------------------------------------------------


def _fr(v) -> Fraction:
    return Fraction(v)


def _as_glyph(raw_contours):
    """Accept either one contour ([{seg}...]) or a glyph ([[{seg}...]])."""
    if raw_contours and isinstance(raw_contours[0], dict):
        return [raw_contours]
    return raw_contours


def _line_edges(raw_contours):
    edges = []
    for contour in _as_glyph(raw_contours):
        for seg in contour:
            if seg["type"] != "line":
                raise TypeError("line-only reference got a non-line segment")
            edges.append(
                (
                    _fr(seg["p0"]["x"]),
                    _fr(seg["p0"]["y"]),
                    _fr(seg["p1"]["x"]),
                    _fr(seg["p1"]["y"]),
                )
            )
    return edges


def winding_point_lines(raw_contours, sx: Fraction, sy: Fraction) -> int:
    """Exact signed winding of one point via per-edge ray casting."""
    total = 0
    for x0, y0, x1, y1 in _line_edges(raw_contours):
        span = _span_and_dir(y0, y1)
        if span is None:
            continue
        lo, hi, direction = span
        if lo <= sy < hi:
            x = x0 + (sy - y0) * (x1 - x0) / (y1 - y0)
            if x >= sx:
                total += direction
    return total


def reference_levels_lines(raw_contours, width: int, height: int):
    """Return (levels, windings) with exact Fraction arithmetic."""
    edges = _line_edges(raw_contours)
    rows_by_py = {}
    for py in range(height):
        per_j = {}
        for j in range(4):
            sy = Fraction(py) + _SAMPLE_OFFSETS[j]
            hits = []
            for x0, y0, x1, y1 in edges:
                span = _span_and_dir(y0, y1)
                if span is None:
                    continue
                lo, hi, direction = span
                if lo <= sy < hi:
                    x = x0 + (sy - y0) * (x1 - x0) / (y1 - y0)
                    hits.append((x, direction))
            per_j[j] = hits
        rows_by_py[py] = per_j
    return _assemble(rows_by_py, width, height)


# ---------------------------------------------------------------------------
# Decimal reference: lines + quadratic Beziers, bisection roots
# ---------------------------------------------------------------------------

_D = Decimal
_ZERO = _D(0)
_ONE = _D(1)
_TWO = _D(2)


def _dec(v) -> Decimal:
    # Route through Fraction so binary floats (e.g. 0.125) convert exactly.
    return Decimal(Fraction(v).numerator) / Decimal(Fraction(v).denominator)


def _quad_y(t, y0, cy, y1):
    mt = _ONE - t
    return mt * mt * y0 + _TWO * mt * t * cy + t * t * y1


def _quad_x(t, x0, cx, x1):
    mt = _ONE - t
    return mt * mt * x0 + _TWO * mt * t * cx + t * t * x1


def _y_extremum_parameter(y0, cy, y1):
    """Interior stationary parameter t* = (y0-cy)/(y0-2cy+y1), or None."""
    denom = y0 - _TWO * cy + y1
    if denom == _ZERO:
        return None
    t = (y0 - cy) / denom
    if _ZERO < t < _ONE:
        return t
    return None


def _bisect_root(y0, cy, y1, ta, tb, sy):
    """Root of y(t)=sy on a monotonic piece; None at the excluded top end."""
    ya = _quad_y(ta, y0, cy, y1)
    yb = _quad_y(tb, y0, cy, y1)
    span = _span_and_dir(ya, yb)
    if span is None:
        return None
    lo, hi, _ = span
    if not (lo <= sy < hi):
        return None
    if sy == ya:
        return ta
    up = yb > ya

    def f(t):
        v = _quad_y(t, y0, cy, y1) - sy
        return v if up else -v

    left, right = ta, tb
    for _ in range(_BISECTION_STEPS):
        mid = (left + right) / _TWO
        if f(mid) < _ZERO:
            left = mid
        else:
            right = mid
    return (left + right) / _TWO


def _pieces_decimal(raw_contours):
    """Flatten all contours into y-monotonic Decimal pieces."""
    out = []
    for contour in _as_glyph(raw_contours):
        for seg in contour:
            if seg["type"] == "line":
                out.append(
                    (
                        "line",
                        _dec(seg["p0"]["x"]), _dec(seg["p0"]["y"]),
                        _dec(seg["p1"]["x"]), _dec(seg["p1"]["y"]),
                    )
                )
            else:
                x0, y0 = _dec(seg["p0"]["x"]), _dec(seg["p0"]["y"])
                cx, cy = _dec(seg["c"]["x"]), _dec(seg["c"]["y"])
                x1, y1 = _dec(seg["p1"]["x"]), _dec(seg["p1"]["y"])
                tstar = _y_extremum_parameter(y0, cy, y1)
                cuts = (
                    [_ZERO, _ONE] if tstar is None else [_ZERO, tstar, _ONE]
                )
                for k in range(len(cuts) - 1):
                    out.append(
                        (
                            "curve", x0, y0, cx, cy, x1, y1,
                            cuts[k], cuts[k + 1],
                        )
                    )
    return out


def winding_point_mixed(raw_contours, sx: Decimal, sy: Decimal) -> int:
    """Signed winding of one point (Decimal) over lines + quads."""
    total = 0
    for piece in _pieces_decimal(raw_contours):
        if piece[0] == "line":
            _, x0, y0, x1, y1 = piece
            span = _span_and_dir(y0, y1)
            if span is None:
                continue
            lo, hi, direction = span
            if lo <= sy < hi:
                x = x0 + (sy - y0) * (x1 - x0) / (y1 - y0)
                if x >= sx:
                    total += direction
        else:
            _, x0, y0, cx, cy, x1, y1, ta, tb = piece
            t = _bisect_root(y0, cy, y1, ta, tb, sy)
            if t is not None:
                ya = _quad_y(ta, y0, cy, y1)
                yb = _quad_y(tb, y0, cy, y1)
                direction = 1 if yb > ya else -1
                if _quad_x(t, x0, cx, x1) >= sx:
                    total += direction
    return total


def reference_levels_mixed(raw_contours, width: int, height: int):
    """Return (levels, windings) via 60-digit Decimal bisection."""
    pieces = _pieces_decimal(raw_contours)
    rows_by_py = {}
    for py in range(height):
        per_j = {}
        for j in range(4):
            sy = Decimal(py) + _SAMPLE_OFFSETS_D[j]
            hits = []
            for piece in pieces:
                if piece[0] == "line":
                    _, x0, y0, x1, y1 = piece
                    span = _span_and_dir(y0, y1)
                    if span is None:
                        continue
                    lo, hi, direction = span
                    if lo <= sy < hi:
                        x = x0 + (sy - y0) * (x1 - x0) / (y1 - y0)
                        hits.append((x, direction))
                else:
                    _, x0, y0, cx, cy, x1, y1, ta, tb = piece
                    t = _bisect_root(y0, cy, y1, ta, tb, sy)
                    if t is not None:
                        ya = _quad_y(ta, y0, cy, y1)
                        yb = _quad_y(tb, y0, cy, y1)
                        direction = 1 if yb > ya else -1
                        hits.append((_quad_x(t, x0, cx, x1), direction))
            per_j[j] = hits
        rows_by_py[py] = per_j
    return _assemble(rows_by_py, width, height, decimal=True)


# ---------------------------------------------------------------------------
# Exact clipped nonzero-winding area (whole-window reference)
# ---------------------------------------------------------------------------
#
# Independent of winding_fill.area: instead of clipping edges into each
# pixel, the whole window rectangle is decomposed into vertical slabs at
# every critical x (segment endpoints + pairwise intersections), each
# slab is cut into bands by the edges crossing it, and bands whose
# midpoint has nonzero winding contribute their exact trapezoid area.
# Everything is fractions.Fraction arithmetic.


def _ref_segments(polygons):
    segs = []
    for poly in polygons:
        n = len(poly)
        for i in range(n):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % n]
            a = (Fraction(x0), Fraction(y0))
            b = (Fraction(x1), Fraction(y1))
            if a != b:
                segs.append((a, b))
    return segs


def _ref_winding(segs, x, y):
    winding = 0
    for (ax, ay), (bx, by) in segs:
        span = _span_and_dir(ay, by)
        if span is None:
            continue
        lo, hi, direction = span
        if lo <= y < hi:
            xc = ax + (y - ay) * (bx - ax) / (by - ay)
            if xc >= x:
                winding += direction
    return winding


def _ref_intersect_x(s1, s2):
    (ax, ay), (bx, by) = s1
    (cx, cy), (dx, dy) = s2
    rx, ry = bx - ax, by - ay
    sx, sy = dx - cx, dy - cy
    den = rx * sy - ry * sx
    if den == 0:
        return None
    t = ((cx - ax) * sy - (cy - ay) * sx) / den
    u = ((cx - ax) * ry - (cy - ay) * rx) / den
    if 0 < t < 1 and 0 < u < 1:
        return ax + t * rx
    return None


def reference_clipped_area(polygons, left, bottom, width, height):
    """Exact area of the nonzero-winding region inside a window rect.

    polygons: list of vertex rings, each a list of (x, y) number pairs
    (implicitly closed).  The window is the rectangle
    [left, left+width] x [bottom, bottom+height].  Returns a Fraction.
    """
    segs = _ref_segments(polygons)
    x0, x1 = Fraction(left), Fraction(left + width)
    y0, y1 = Fraction(bottom), Fraction(bottom + height)

    critical = {x0, x1}
    for (ax, ay), (bx, by) in segs:
        for v in (ax, bx):
            if x0 < v < x1:
                critical.add(v)
        # x where the segment crosses the window's horizontal borders:
        # the band structure changes there as well.
        if ay != by:
            for yb in (y0, y1):
                lo, hi = (ay, by) if ay < by else (by, ay)
                if lo < yb < hi:
                    v = ax + (yb - ay) * (bx - ax) / (by - ay)
                    if x0 < v < x1:
                        critical.add(v)
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            v = _ref_intersect_x(segs[i], segs[j])
            if v is not None and x0 < v < x1:
                critical.add(v)
    xs = sorted(critical)

    def y_at(seg, x):
        (ax, ay), (bx, by) = seg
        return ay + (x - ax) * (by - ay) / (bx - ax)

    total = Fraction(0)
    for xa, xb in zip(xs, xs[1:]):
        if xa == xb:
            continue
        xm = (xa + xb) / 2
        # y values where edges cross the vertical line x = xm, clipped
        # to the window's y range.
        ys = [y0, y1]
        for seg in segs:
            (ax, _), (bx, _) = seg
            if ax == bx:
                continue
            lo, hi = (ax, bx) if ax < bx else (bx, ax)
            if lo <= xm <= hi:
                v = y_at(seg, xm)
                if y0 < v < y1:
                    ys.append(v)
        ys = sorted(set(ys))
        for ylo, yhi in zip(ys, ys[1:]):
            ymid = (ylo + yhi) / 2
            if _ref_winding(segs, xm, ymid) != 0:
                total += (xb - xa) * (yhi - ylo)
    return total
