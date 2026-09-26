"""Exact rational grayscale area coverage for polygon contours.

This module adds an exact-area entry point on top of the nonzero-winding
model of :mod:`winding_fill.core`.  Instead of the fixed 4x4 sub-sample
approximation, every pixel value here is the *exact* area of the
nonzero-winding region inside that pixel's unit square, computed with
:class:`fractions.Fraction` arithmetic -- no supersampling, no tolerance.

Method (implemented from scratch, no font/raster library used):

1. Coordinates are converted to exact rationals (the exact binary value
   of each accepted int/float).
2. For each pixel of the requested window, every polygon edge is clipped
   to the pixel's unit square (exact Liang-Barsky clipping).
3. Inside the pixel the clipped edges form an arrangement.  Vertical
   slab boundaries are the pixel sides, the clipped-edge endpoints and
   all pairwise edge crossings.  Within one slab edges never cross, so
   the slab is cut into facets between consecutive edges; each facet's
   signed winding number is sampled once at its midpoint by an exact
   ray cast, and facets with nonzero winding contribute their exact
   trapezoid area.  A region wound twice is still one facet, so it is
   counted once -- never twice as bright.
4. Edges lying exactly on pixel boundaries contribute only through the
   areas they bound; there is no extra line width.

Only straight line segments are accepted (polygon contours).  Quadratic
Bezier segments are rejected with :class:`InvalidInputError`; use
:func:`winding_fill.rasterize` for curved input.

Coordinate space matches the rest of the package: x right, y up, pixel
``(px, py)`` covers ``[px, px+1] x [py, py+1]`` and row 0 is the bottom
row of the window.  The window origin may be negative.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Sequence, Tuple

from .errors import InvalidInputError
from .geometry import MAX_CONTOURS, MAX_COORD_ABS, Contour, Line, canvas_dims

_ZERO = Fraction(0)
_ONE = Fraction(1)
_HALF = Fraction(1, 2)


def _validate_origin(origin: object) -> Tuple[int, int]:
    if not isinstance(origin, (list, tuple)) or len(origin) != 2:
        raise InvalidInputError(
            "origin", "origin must be an (ox, oy) pair of integers"
        )
    out: List[int] = []
    for name, v in (("origin.x", origin[0]), ("origin.y", origin[1])):
        if isinstance(v, bool) or not isinstance(v, int):
            raise InvalidInputError(
                name, f"expected integer, got {type(v).__name__}"
            )
        if abs(v) > MAX_COORD_ABS:
            raise InvalidInputError(
                name, f"origin {v} exceeds limit |x| <= {MAX_COORD_ABS}"
            )
        out.append(int(v))
    return out[0], out[1]


def _contours_to_edges(
    contours: Sequence[Contour],
) -> List[Tuple[Fraction, Fraction, Fraction, Fraction]]:
    edges: List[Tuple[Fraction, Fraction, Fraction, Fraction]] = []
    for ci, con in enumerate(contours):
        for si, seg in enumerate(con.segments):
            if not isinstance(seg, Line):
                raise InvalidInputError(
                    f"contours[{ci}].segments[{si}]",
                    "exact-area rasterizer accepts line segments only "
                    "(polygon contours); use rasterize() for quadratic "
                    "curves",
                )
            edges.append(
                (
                    Fraction(seg.p0.x),
                    Fraction(seg.p0.y),
                    Fraction(seg.p1.x),
                    Fraction(seg.p1.y),
                )
            )
    return edges


def _winding_at(edges, sx: Fraction, sy: Fraction) -> int:
    """Exact signed winding number at ``(sx, sy)`` (ray to +x infinity).

    Half-open y rule ``lo <= sy < hi``; horizontal edges never count.
    The sample points chosen by the rasterizer never lie on an edge, so
    the ``xc >= sx`` comparison is never a tie in practice.
    """
    winding = 0
    for x0, y0, x1, y1 in edges:
        if y0 == y1:
            continue
        lo, hi = (y0, y1) if y0 < y1 else (y1, y0)
        if not (lo <= sy < hi):
            continue
        xc = x0 + (sy - y0) * (x1 - x0) / (y1 - y0)
        if xc >= sx:
            winding += 1 if y1 > y0 else -1
    return winding


def _clip_edge(x0, y0, x1, y1, xmin, ymin, xmax, ymax):
    """Exact Liang-Barsky clip of one edge to the pixel box."""
    dx = x1 - x0
    dy = y1 - y0
    t0, t1 = _ZERO, _ONE
    for p, q in (
        (-dx, x0 - xmin),
        (dx, xmax - x0),
        (-dy, y0 - ymin),
        (dy, ymax - y0),
    ):
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
    return (x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy)


def _edge_crossing_x(e1, e2):
    """x coordinate of the intersection of two clipped edges, if any."""
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


def _pixel_area(edges, px: int, py: int) -> Fraction:
    """Exact nonzero-winding area inside pixel ``(px, py)``."""
    xmin, ymin = Fraction(px), Fraction(py)
    xmax, ymax = xmin + 1, ymin + 1
    clipped = []
    for edge in edges:
        c = _clip_edge(*edge, xmin, ymin, xmax, ymax)
        if c is not None:
            clipped.append(c)
    if not clipped:
        winding = _winding_at(edges, xmin + _HALF, ymin + _HALF)
        return _ONE if winding != 0 else _ZERO

    # Slab boundaries: pixel sides, clipped endpoints, edge crossings.
    xs = {xmin, xmax}
    for ax, ay, bx, by in clipped:
        for x in (ax, bx):
            if xmin < x < xmax:
                xs.add(x)
    for i in range(len(clipped)):
        for j in range(i + 1, len(clipped)):
            x = _edge_crossing_x(clipped[i], clipped[j])
            if x is not None and xmin < x < xmax:
                xs.add(x)
    xs = sorted(xs)

    total = _ZERO
    for xa, xb in zip(xs, xs[1:]):
        mid = (xa + xb) / 2
        # Edges spanning the whole slab, bottom to top: (y_mid, y_xa, y_xb).
        bounds = []
        for ax, ay, bx, by in clipped:
            if ax == bx:
                continue  # vertical edges bound no slab
            lo_x, hi_x = (ax, bx) if ax < bx else (bx, ax)
            if lo_x <= xa and hi_x >= xb:
                ya = ay + (by - ay) * (xa - ax) / (bx - ax)
                yb = ay + (by - ay) * (xb - ax) / (bx - ax)
                bounds.append(((ya + yb) / 2, ya, yb))
        bounds.sort(key=lambda b: b[0])
        bounds = [(ymin, ymin, ymin)] + bounds + [(ymax, ymax, ymax)]
        for lo, hi in zip(bounds, bounds[1:]):
            if hi[0] <= lo[0]:
                continue  # empty facet (duplicate/touching edges)
            sy = (lo[0] + hi[0]) / 2
            if _winding_at(edges, mid, sy) != 0:
                total += (xb - xa) * ((hi[1] - lo[1]) + (hi[2] - lo[2])) / 2
    return total


@dataclass(frozen=True)
class AreaCoverageResult:
    """Exact grayscale coverage for one window.

    Attributes:
        origin_x, origin_y: integer window origin (may be negative);
        width, height: window size in pixels;
        areas: row-major tuple (row 0 is the bottom) of tuples of
            :class:`fractions.Fraction` -- the exact nonzero-winding
            area inside each pixel's unit square, always in ``[0, 1]``.
    """

    origin_x: int
    origin_y: int
    width: int
    height: int
    areas: Tuple[Tuple[Fraction, ...], ...]

    def _check_pixel(self, px: int, py: int) -> None:
        for name, v, hi in (
            ("px", px, self.width),
            ("py", py, self.height),
        ):
            if isinstance(v, bool) or not isinstance(v, int):
                raise InvalidInputError(
                    name, f"expected integer, got {type(v).__name__}"
                )
            if not 0 <= v < hi:
                raise InvalidInputError(name, f"{v} outside 0..{hi - 1}")

    def area_at(self, px: int, py: int) -> Fraction:
        """Exact covered area of window pixel ``(px, py)`` in ``[0, 1]``."""
        self._check_pixel(px, py)
        return self.areas[py][px]

    def coverage_at(self, px: int, py: int) -> float:
        """Float view of :meth:`area_at` (exact value converted to float)."""
        return float(self.area_at(px, py))

    def total_area(self) -> Fraction:
        """Exact sum of all pixel areas in the window."""
        return Fraction(sum(sum(row) for row in self.areas))


def _validate_contours(contours: object) -> Sequence[Contour]:
    if not isinstance(contours, (list, tuple)):
        raise InvalidInputError(
            "contours",
            "expected a list/tuple of Contour objects; raw data goes "
            "through parse_contours() or rasterize_area_raw()",
        )
    if len(contours) == 0:
        raise InvalidInputError("contours", "at least one contour is required")
    if len(contours) > MAX_CONTOURS:
        raise InvalidInputError(
            "contours", f"too many contours: {len(contours)} > {MAX_CONTOURS}"
        )
    for ci, con in enumerate(contours):
        if not isinstance(con, Contour):
            raise InvalidInputError(
                f"contours[{ci}]",
                f"expected Contour, got {type(con).__name__}",
            )
    return contours


def rasterize_area(
    contours: Sequence[Contour],
    width: int,
    height: int,
    *,
    origin: object = (0, 0),
) -> AreaCoverageResult:
    """Exact-area rasterization of polygon contours into a small window.

    ``contours`` must be validated line-only :class:`Contour` objects
    (1..``MAX_CONTOURS``).  ``width``/``height`` are integers in
    ``1..MAX_CANVAS`` (``bool`` rejected); ``origin`` is an integer
    ``(ox, oy)`` pair, possibly negative, placing the window's
    bottom-left pixel.  Pixel ``(px, py)`` of the result covers the unit
    square ``[ox+px, ox+px+1] x [oy+py, oy+py+1]``.

    Every pixel value is the exact rational area of the nonzero-winding
    region inside that unit square; regions wound more than once count
    once.  Quadratic Bezier segments are rejected.

    Raises:
        InvalidInputError: bad window, bad origin, non-line segments or
            an invalid contour sequence.
    """
    width, height = canvas_dims(width, height)
    ox, oy = _validate_origin(origin)
    if ox + width > MAX_COORD_ABS or oy + height > MAX_COORD_ABS:
        raise InvalidInputError(
            "origin",
            f"window extends beyond coordinate limit {MAX_COORD_ABS}",
        )
    contours = _validate_contours(contours)
    edges = _contours_to_edges(contours)

    rows: List[Tuple[Fraction, ...]] = []
    for py in range(height):
        row = []
        for px in range(width):
            row.append(_pixel_area(edges, ox + px, oy + py))
        rows.append(tuple(row))
    return AreaCoverageResult(
        origin_x=ox,
        origin_y=oy,
        width=width,
        height=height,
        areas=tuple(rows),
    )


def rasterize_area_raw(
    raw_contours: object,
    width: int,
    height: int,
    *,
    origin: object = (0, 0),
) -> AreaCoverageResult:
    """Validate raw dict geometry and exact-area rasterize it in one call.

    See :func:`winding_fill.geometry.parse_contours` for the accepted
    shape.  Validation happens before any rasterization, so invalid
    input never produces a partial result.
    """
    from .geometry import parse_contours

    contours = parse_contours(raw_contours)  # raises on the first bad value
    return rasterize_area(contours, width, height, origin=origin)
