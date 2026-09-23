"""Core nonzero-winding rasterizer.

Algorithm (implemented from scratch, no font/raster library used):

1. Every segment is converted into one or more **y-monotonic pieces**.
   Lines are already monotonic (or horizontal).  A quadratic Bézier
   ``B(t)`` with ``y(t) = a t^2 + b t + y0`` is split at the interior
   parameter ``t* = -b / (2a)`` where ``y'(t*) = 0``, so each resulting
   piece is strictly monotonic in y or horizontal.

2. For each of the fixed 4x4 sub-sample rows, a horizontal ray is cast and
   every piece crossing the row's y contributes **one signed** crossing:

   * upward piece (``y_end > y_start``): +1
   * downward piece: -1
   * horizontal pieces: 0 (never counted)

   The *half-open y rule* decides whether a row at ``sy`` meets a piece
   with endpoint levels ``lo < hi``::

       lo <= sy < hi        # count exactly once

   so a vertex shared by two edges is counted by exactly one of them.
   At a quadratic y-extremum the same precomputed float extremum value is
   used for both adjacent pieces: either both pieces contribute (a
   cancelling +1/-1 pair, the tangent case) or neither does -- the
   decision is always deterministic, never tolerance-driven.

3. Crossing x is found by solving ``y(t) = sy`` (linear interpolation for
   lines, the quadratic formula for curves) on the monotonic piece.

4. Crossings are sorted by x.  For a sub-sample point, the winding number
   is the signed sum of all crossings at ``x >= sample_x`` (a sample
   exactly on an edge is treated as inside).  Coverage is nonzero when
   the winding number is non-zero.

Coordinate space: x right, **y up**.  Pixel ``(px, py)`` covers the unit
square ``[px, px+1] x [py, py+1]``; grid row 0 is therefore the bottom
row.  Sub-sample ``(i, j)`` (both 0..3) sits at
``(px + (2i+1)/8, py + (2j+1)/8)``.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .errors import InvalidInputError
from .geometry import MAX_CONTOURS, Contour, Line, Quad

SUBSAMPLES = 4
"""Fixed sub-sampling: 4 x 4 = 16 samples per pixel (part of contract)."""

_SAMPLE_CENTER = 0.125  # (2*0+1)/8
_SAMPLE_STEP = 0.25

# Internal piece tags
_LINE = 0
_CURVE = 1


@dataclass(frozen=True)
class _Piece:
    """A y-monotonic piece of one segment.

    Lines: kind=_LINE, (x0,y0)->(x1,y1), curve fields unused.
    Curves: kind=_CURVE, B(t) over [ta, tb] with
            y(t) = a t^2 + b t + cy0, endpoint y values ya=y(ta), yb=y(tb).
    """

    kind: int
    ya: float
    yb: float
    # line geometry
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    # curve parameter range / polynomial coefficients
    ta: float = 0.0
    tb: float = 1.0
    a: float = 0.0
    b: float = 0.0
    cy0: float = 0.0
    cx0: float = 0.0
    cx: float = 0.0
    cx1: float = 0.0


def _line_piece(seg: Line) -> _Piece:
    return _Piece(
        kind=_LINE,
        ya=seg.p0.y,
        yb=seg.p1.y,
        x0=seg.p0.x,
        y0=seg.p0.y,
        x1=seg.p1.x,
        y1=seg.p1.y,
    )


def _curve_pieces(seg: Quad) -> List[_Piece]:
    """Split a quadratic Bézier into y-monotonic pieces."""
    x0, y0 = seg.p0.x, seg.p0.y
    cx, cy = seg.c.x, seg.c.y
    x1, y1 = seg.p1.x, seg.p1.y
    a = y0 - 2.0 * cy + y1
    b = 2.0 * (cy - y0)

    # Interior y-extremum parameter of the parabola, if any.
    cuts = [0.0, 1.0]
    if a != 0.0:
        t = -b / (2.0 * a)
        if 0.0 < t < 1.0:
            cuts = [0.0, t, 1.0]

    def yval(t: float) -> float:
        mt = 1.0 - t
        return mt * mt * y0 + 2.0 * mt * t * cy + t * t * y1

    ycuts = [yval(t) for t in cuts]

    pieces: List[_Piece] = []
    for k in range(len(cuts) - 1):
        ta, tb = cuts[k], cuts[k + 1]
        pieces.append(
            _Piece(
                kind=_CURVE,
                ya=ycuts[k],
                yb=ycuts[k + 1],
                ta=ta,
                tb=tb,
                a=a,
                b=b,
                cy0=y0,
                cx0=x0,
                cx=cx,
                cx1=x1,
            )
        )
    return pieces


def _build_pieces(contours: Sequence[Contour]) -> List[_Piece]:
    pieces: List[_Piece] = []
    for con in contours:
        for seg in con.segments:
            if isinstance(seg, Line):
                pieces.append(_line_piece(seg))
            else:
                assert isinstance(seg, Quad)
                pieces.extend(_curve_pieces(seg))
    return pieces


def _crossing(p: _Piece, sy: float) -> Tuple[float, int] | None:
    """Return ``(x, direction)`` if this monotonic piece crosses row ``sy``.

    Direction is +1 for upward traversal, -1 downward; horizontal pieces
    never cross.  The half-open interval ``lo <= sy < hi`` makes a vertex
    sitting exactly on a ray count through exactly one adjacent piece.
    """
    ya, yb = p.ya, p.yb
    if yb > ya:
        direction = 1
        lo, hi = ya, yb
    elif yb < ya:
        direction = -1
        lo, hi = yb, ya
    else:
        return None  # horizontal piece: explicitly not counted
    if not (lo <= sy < hi):
        return None

    if p.kind == _LINE:
        t = (sy - p.y0) / (p.y1 - p.y0)
        x = p.x0 + t * (p.x1 - p.x0)
        return x, direction

    # Quadratic: solve y(t) = a t^2 + b t + cy0 = sy on [ta, tb].
    if sy == ya:
        # Shared split endpoint: the same float extremum is visible to
        # both adjacent pieces, so tangent handling is deterministic.
        t = p.ta
    elif p.a == 0.0:
        t = (sy - p.cy0) / p.b
    else:
        disc = p.b * p.b - 4.0 * p.a * (p.cy0 - sy)
        if disc < 0.0:
            # sy can sit a round-off beyond the true extremum only when
            # it was computed from y(ta)/y(tb); clamp to tangency.
            disc = 0.0
        s = math.sqrt(disc)
        t_lo = (-p.b - s) / (2.0 * p.a)
        t_hi = (-p.b + s) / (2.0 * p.a)
        if p.ta <= t_lo <= p.tb:
            t = t_lo
        elif p.ta <= t_hi <= p.tb:
            t = t_hi
        else:  # pragma: no cover - defensive round-off guard
            t = t_lo if abs(t_lo - p.ta) < abs(t_hi - p.ta) else t_hi
            t = min(max(t, p.ta), p.tb)
    mt = 1.0 - t
    x = mt * mt * p.cx0 + 2.0 * mt * t * p.cx + t * t * p.cx1
    return x, direction


@dataclass(frozen=True)
class CoverageResult:
    """Rasterized output for one canvas.

    Attributes:
        width, height: canvas size in pixels;
        levels: row-major tuple (grid row ``py`` first; row 0 is the
            bottom) of tuples of integers ``0..16`` -- how many of the 16
            sub-samples are covered (nonzero winding);
        windings: for each pixel a 16-tuple of signed winding numbers,
            indexed ``j * 4 + i`` for sub-sample column ``i``, row ``j``.
    """

    width: int
    height: int
    levels: Tuple[Tuple[int, ...], ...]
    windings: Tuple[Tuple[Tuple[int, ...], ...], ...]

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

    def coverage_at(self, px: int, py: int) -> float:
        """Gray level of pixel ``(px, py)``: ``k/16`` with ``k`` in 0..16."""
        self._check_pixel(px, py)
        return self.levels[py][px] / float(SUBSAMPLES * SUBSAMPLES)

    def windings_at(self, px: int, py: int) -> Tuple[int, ...]:
        """16 signed sub-sample winding numbers for pixel ``(px, py)``.

        Index order is sub-row ``j`` first then sub-column ``i``:
        ``index = j * 4 + i``.
        """
        self._check_pixel(px, py)
        return self.windings[py][px]

    def winding_at(self, px: int, py: int, i: int, j: int) -> int:
        """Signed winding number of one sub-sample (``i, j`` in 0..3)."""
        self._check_pixel(px, py)
        for name, v in (("i", i), ("j", j)):
            if isinstance(v, bool) or not isinstance(v, int):
                raise InvalidInputError(
                    name, f"expected integer 0..3, got {type(v).__name__}"
                )
            if not 0 <= v < SUBSAMPLES:
                raise InvalidInputError(name, f"{v} outside 0..3")
        return self.windings[py][px][j * SUBSAMPLES + i]


def rasterize(
    contours: Sequence[Contour], width: int, height: int
) -> CoverageResult:
    """Rasterize closed contours onto a ``width`` x ``height`` canvas.

    ``contours`` must be validated :class:`Contour` objects (use
    :func:`winding_fill.geometry.parse_contours` for raw dict data, or
    :func:`rasterize_raw` for the one-step convenience wrapper).  There
    must be 1..``MAX_CONTOURS`` contours and each canvas side must be an
    integer in ``1..MAX_CANVAS`` (``bool`` rejected).

    Coverage uses the nonzero-winding rule with fixed 4x4 sub-sampling;
    see the module docstring for exact boundary semantics.

    Raises:
        InvalidInputError: bad canvas size, wrong argument types or an
            invalid contour sequence.
    """
    from .geometry import canvas_dims

    width, height = canvas_dims(width, height)
    if not isinstance(contours, (list, tuple)):
        raise InvalidInputError(
            "contours",
            "expected a list/tuple of Contour objects; raw data goes "
            "through parse_contours() or rasterize_raw()",
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

    pieces = _build_pieces(contours)
    return _rasterize_pieces(pieces, width, height)


def rasterize_raw(
    raw_contours: object, width: int, height: int
) -> CoverageResult:
    """Validate raw dict/tuple geometry and rasterize it in one call.

    See :func:`winding_fill.geometry.parse_contours` for the accepted
    shape.  Validation happens before any rasterization, so an invalid
    contour list never produces a partial result.
    """
    from .geometry import parse_contours

    contours = parse_contours(raw_contours)  # raises on the first bad value
    return rasterize(contours, width, height)


def _rasterize_pieces(
    pieces: List[_Piece], width: int, height: int
) -> CoverageResult:
    n = SUBSAMPLES
    levels_rows: List[Tuple[int, ...]] = []
    wind_rows: List[Tuple[Tuple[int, ...], ...]] = []

    for py in range(height):
        winding_by_px: List[List[int]] = [
            [0] * (n * n) for _ in range(width)
        ]
        for j in range(n):
            sy = py + _SAMPLE_CENTER + j * _SAMPLE_STEP
            crossings: List[Tuple[float, int]] = []
            for p in pieces:
                hit = _crossing(p, sy)
                if hit is not None:
                    crossings.append(hit)
            crossings.sort(key=lambda h: h[0])
            xs = [h[0] for h in crossings]
            dirs = [h[1] for h in crossings]
            # Suffix signed sums: winding at sx = sum(dirs[k:]) for the
            # first crossing with x >= sx.
            suffix = [0] * (len(dirs) + 1)
            for k in range(len(dirs) - 1, -1, -1):
                suffix[k] = suffix[k + 1] + dirs[k]
            for px in range(width):
                base = j * n
                for i in range(n):
                    sx = px + _SAMPLE_CENTER + i * _SAMPLE_STEP
                    k = bisect.bisect_left(xs, sx)
                    winding_by_px[px][base + i] = suffix[k]

        levels_rows.append(
            tuple(
                sum(1 for w in winding_by_px[px] if w != 0)
                for px in range(width)
            )
        )
        wind_rows.append(
            tuple(tuple(winding_by_px[px]) for px in range(width))
        )

    return CoverageResult(
        width=width,
        height=height,
        levels=tuple(levels_rows),
        windings=tuple(wind_rows),
    )
