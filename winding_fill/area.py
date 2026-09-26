"""Exact rational gray-level area coverage for polygon contours.

This module adds the exact-area entry point on top of the 4x4
sub-sampling rasterizer in winding_fill.core: the gray value of a
pixel is the *true area* of the nonzero-winding region inside that
pixel's unit square, computed with fractions.Fraction arithmetic
only -- no fixed-rate super-sampling approximation, and a region
wound twice still counts once (nonzero winding, not summed winding).

Inputs may be self-intersecting contours and several mutually
overlapping contours.  Edges lying exactly on pixel boundaries
contribute only through the area they enclose -- there is no extra
stroke/line width.

Algorithm (from scratch, exact rational arithmetic):

1. Every (line) segment is clipped to the pixel square
   [px, px+1] x [py, py+1] with rational Liang-Barsky clipping.
2. Inside one pixel the clipped edges form an arrangement.  All
   critical x coordinates -- clipped endpoints and every exact
   edge/edge intersection -- split the pixel into vertical slabs.  No
   two edges cross inside a slab, so the vertical order of edges is
   constant there.
3. Each slab is cut into horizontal faces (bands) by the edge curves
   and the pixel's top/bottom borders.  The winding number of a band is
   sampled exactly at its midpoint with the same half-open-y horizontal
   ray rule as the sub-sample rasterizer; bands with nonzero winding
   contribute their exact trapezoid area.

Only polygon (straight line) contours are accepted: quadratic Bezier
curves would make pixel-boundary intersections irrational, which is
outside this module's exact-rational contract.

Coordinate space matches the rest of the package: x right, y up, pixel
(px, py) covers [px, px+1] x [py, py+1]; negative pixel coordinates
are fine.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Sequence, Tuple

from .errors import InvalidInputError
from .geometry import (
    MAX_CONTOURS,
    MAX_COORD_ABS,
    Contour,
    Line,
    Quad,
    canvas_dims,
)

__all__ = ["AreaResult", "fill_area", "fill_area_raw"]

# A rational 2-D point.
_Pt = Tuple[Fraction, Fraction]
# A directed rational segment.
_Seg = Tuple[_Pt, _Pt]


def _to_fraction(v: object, where: str) -> Fraction:
    if isinstance(v, Fraction):
        return v
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise InvalidInputError(
            where, f"expected a real number, got {type(v).__name__}"
        )
    return Fraction(v)


def _window(
    left: object, bottom: object, width: object, height: object
) -> Tuple[int, int, int, int]:
    """Validate the output window; returns integer (left, bottom, w, h)."""
    width, height = canvas_dims(width, height)
    out = []
    for name, v in (("left", left), ("bottom", bottom)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise InvalidInputError(
                name,
                f"expected an integer pixel coordinate, got "
                f"{type(v).__name__}",
            )
        out.append(int(v))
    left_i, bottom_i = out
    for name, v in (
        ("left", left_i),
        ("bottom", bottom_i),
        ("left+width", left_i + width),
        ("bottom+height", bottom_i + height),
    ):
        if abs(v) > MAX_COORD_ABS:
            raise InvalidInputError(
                name, f"window corner {v} exceeds limit |x| <= {MAX_COORD_ABS}"
            )
    return left_i, bottom_i, width, height


def _validate_polygon_contours(contours: object) -> List[Contour]:
    if not isinstance(contours, (list, tuple)):
        raise InvalidInputError(
            "contours",
            "expected a list/tuple of Contour objects; raw data goes "
            "through parse_contours() or fill_area_raw()",
        )
    if len(contours) == 0:
        raise InvalidInputError("contours", "at least one contour is required")
    if len(contours) > MAX_CONTOURS:
        raise InvalidInputError(
            "contours", f"too many contours: {len(contours)} > {MAX_CONTOURS}"
        )
    out: List[Contour] = []
    for ci, con in enumerate(contours):
        if not isinstance(con, Contour):
            raise InvalidInputError(
                f"contours[{ci}]", f"expected Contour, got {type(con).__name__}"
            )
        for si, seg in enumerate(con.segments):
            if isinstance(seg, Quad):
                raise InvalidInputError(
                    f"contours[{ci}].segments[{si}]",
                    "exact area fill accepts polygon (line) contours only; "
                    "quadratic curves are not exact-rational representable "
                    "against pixel boundaries",
                )
            if not isinstance(seg, Line):
                raise InvalidInputError(
                    f"contours[{ci}].segments[{si}]",
                    f"unknown segment type {type(seg).__name__}",
                )
        out.append(con)
    return out


def _segments_of(contours: Sequence[Contour]) -> List[_Seg]:
    segs: List[_Seg] = []
    for con in contours:
        for seg in con.segments:
            assert isinstance(seg, Line)
            a = (
                _to_fraction(seg.p0.x, "p0.x"),
                _to_fraction(seg.p0.y, "p0.y"),
            )
            b = (
                _to_fraction(seg.p1.x, "p1.x"),
                _to_fraction(seg.p1.y, "p1.y"),
            )
            segs.append((a, b))
    return segs


def _clip_to_square(seg, x0, y0, x1, y1):
    """Liang-Barsky clip of a directed segment to the unit square.

    Returns the clipped directed segment, or None when it misses the
    square.  All arithmetic is exact.
    """
    (ax, ay), (bx, by) = seg
    dx, dy = bx - ax, by - ay
    t0, t1 = Fraction(0), Fraction(1)
    for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return None
            if r < t1:
                t1 = r
    if t0 == t1:
        return None  # touches the square in a single point: zero area
    return ((ax + t0 * dx, ay + t0 * dy), (ax + t1 * dx, ay + t1 * dy))


def _intersection(s1, s2):
    """Exact intersection point of two segments, or None."""
    (ax, ay), (bx, by) = s1
    (cx, cy), (dx, dy) = s2
    rx, ry = bx - ax, by - ay
    sx, sy = dx - cx, dy - cy
    den = rx * sy - ry * sx
    if den == 0:
        return None  # parallel or collinear: no isolated crossing
    t = ((cx - ax) * sy - (cy - ay) * sx) / den
    u = ((cx - ax) * ry - (cy - ay) * rx) / den
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (ax + t * rx, ay + t * ry)
    return None


def _y_at(seg, x):
    """Y of a non-vertical segment at abscissa x (exact)."""
    (ax, ay), (bx, by) = seg
    return ay + (x - ax) * (by - ay) / (bx - ax)


def _winding_at(segs, x, y) -> int:
    """Exact nonzero-rule winding number at point (x, y).

    Horizontal ray to +infinity, half-open y rule (lo <= y < hi),
    upward crossings count +1, downward -1; a point exactly on an edge
    counts as inside (x_cross >= x).  Mirrors the sub-sample
    rasterizer's boundary semantics.
    """
    winding = 0
    for (ax, ay), (bx, by) in segs:
        if ay == by:
            continue  # horizontal edge: never counted
        if ay < by:
            lo, hi, direction = ay, by, 1
        else:
            lo, hi, direction = by, ay, -1
        if lo <= y < hi:
            xc = ax + (y - ay) * (bx - ax) / (by - ay)
            if xc >= x:
                winding += direction
    return winding


def _pixel_area(segs, px: int, py: int) -> Fraction:
    """Exact area of the nonzero-winding region inside one pixel."""
    x0, y0 = Fraction(px), Fraction(py)
    x1, y1 = x0 + 1, y0 + 1

    pieces = []
    for seg in segs:
        clipped = _clip_to_square(seg, x0, y0, x1, y1)
        if clipped is not None:
            pieces.append(clipped)
    if not pieces:
        # No edge crosses the pixel interior: uniformly inside or
        # outside.  Sample the center -- a corner can sit exactly on an
        # edge, where the on-edge rule would misclassify the pixel.
        center = (x0 + Fraction(1, 2), y0 + Fraction(1, 2))
        return Fraction(1) if _winding_at(segs, *center) != 0 else Fraction(0)

    critical = {x0, x1}
    for (a, b) in pieces:
        critical.add(a[0])
        critical.add(b[0])
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            hit = _intersection(pieces[i], pieces[j])
            if hit is not None and x0 < hit[0] < x1:
                critical.add(hit[0])
    xs = sorted(critical)

    total = Fraction(0)
    for xa, xb in zip(xs, xs[1:]):
        if xa == xb:
            continue
        xm = (xa + xb) / 2
        # Boundary curves active across this slab: (kind, payload) where
        # kind 0 = constant y, kind 1 = segment evaluated linearly.
        bounds = [(0, y0)]
        for piece in pieces:
            (ax, _), (bx, _) = piece
            if ax == bx:
                continue  # vertical edge: only a slab separator
            lo, hi = (ax, bx) if ax < bx else (bx, ax)
            if lo <= xm <= hi:
                bounds.append((1, piece))
        bounds.append((0, y1))

        def eval_bound(bound, x):
            kind, payload = bound
            if kind == 0:
                return payload
            return _y_at(payload, x)

        bounds.sort(key=lambda b: eval_bound(b, xm))
        # Merge duplicate boundary curves (coincident edges).
        merged = []
        for b in bounds:
            if merged and eval_bound(merged[-1], xm) == eval_bound(b, xm):
                continue
            merged.append(b)

        for lower, upper in zip(merged, merged[1:]):
            ylo_m = eval_bound(lower, xm)
            yhi_m = eval_bound(upper, xm)
            if yhi_m <= ylo_m:
                continue
            ymid = (ylo_m + yhi_m) / 2
            if _winding_at(segs, xm, ymid) == 0:
                continue
            # Exact trapezoid area between the two boundary curves.
            height_a = eval_bound(upper, xa) - eval_bound(lower, xa)
            height_b = eval_bound(upper, xb) - eval_bound(lower, xb)
            total += (xb - xa) * (height_a + height_b) / 2
    return total


@dataclass(frozen=True)
class AreaResult:
    """Exact gray-level area coverage over one pixel window.

    Attributes:
        left, bottom: integer pixel coordinate of the window's
            lower-left corner (may be negative);
        width, height: window size in pixels;
        areas: row-major tuple areas[row][col] of fractions.Fraction
            values in [0, 1]; row 0 is the *bottom* row of the window
            (y up).
    """

    left: int
    bottom: int
    width: int
    height: int
    areas: Tuple[Tuple[Fraction, ...], ...]

    def _check_pixel(self, px: int, py: int) -> Tuple[int, int]:
        for name, v, lo, hi in (
            ("px", px, self.left, self.left + self.width),
            ("py", py, self.bottom, self.bottom + self.height),
        ):
            if isinstance(v, bool) or not isinstance(v, int):
                raise InvalidInputError(
                    name, f"expected integer, got {type(v).__name__}"
                )
            if not lo <= v < hi:
                raise InvalidInputError(name, f"{v} outside {lo}..{hi - 1}")
        return px - self.left, py - self.bottom

    def area_at(self, px: int, py: int) -> Fraction:
        """Exact covered area of pixel (px, py) (absolute coords)."""
        col, row = self._check_pixel(px, py)
        return self.areas[row][col]

    def gray_at(self, px: int, py: int) -> float:
        """Float gray value of pixel (px, py) in [0.0, 1.0]."""
        return float(self.area_at(px, py))

    def total_area(self) -> Fraction:
        """Sum of all pixel areas in the window (exact)."""
        return Fraction(sum(sum(row) for row in self.areas))


def fill_area(
    contours: Sequence[Contour],
    left: int,
    bottom: int,
    width: int,
    height: int,
) -> AreaResult:
    """Exact rational area coverage of polygon contours over a window.

    contours must be validated Contour objects containing **line
    segments only** (self-intersection and mutual overlap are fine).
    The window covers pixels (left .. left+width-1) x
    (bottom .. bottom+height-1); each side is an integer in
    1..MAX_CANVAS and the origin may be negative.

    The gray value of pixel (px, py) is the exact area of the
    nonzero-winding region inside [px, px+1] x [py, py+1] as a
    fractions.Fraction.  Edges exactly on pixel boundaries contribute
    no extra width.

    Raises:
        InvalidInputError: bad window, wrong argument types, non-line
            segments, or an invalid contour sequence.
    """
    left, bottom, width, height = _window(left, bottom, width, height)
    valid = _validate_polygon_contours(contours)
    segs = _segments_of(valid)

    rows = []
    for py in range(bottom, bottom + height):
        rows.append(
            tuple(_pixel_area(segs, px, py) for px in range(left, left + width))
        )
    return AreaResult(
        left=left,
        bottom=bottom,
        width=width,
        height=height,
        areas=tuple(rows),
    )


def fill_area_raw(
    raw_contours: object,
    left: int,
    bottom: int,
    width: int,
    height: int,
) -> AreaResult:
    """Validate raw dict geometry and compute exact areas in one call.

    Same input input contract as winding_fill.geometry.parse_contours
    plus the polygon-only restriction of fill_area.  Validation happens
    before any computation, so invalid input never produces a partial
    result.
    """
    from .geometry import parse_contours

    contours = parse_contours(raw_contours)  # raises on the first bad value
    return fill_area(contours, left, bottom, width, height)
