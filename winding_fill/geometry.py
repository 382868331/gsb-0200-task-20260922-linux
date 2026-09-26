"""Geometry data structures and strict input validation.

Public, immutable (frozen dataclasses) structures:

* :class:`Point`  -- one rational 2-D coordinate
* :class:`Line`   -- straight segment ``p0 -> p1``
* :class:`Quad`   -- quadratic Bézier with on-curve endpoints and one
  off-curve control point
* :class:`Contour` -- an ordered list of segments forming one closed loop

A contour is *closed* geometrically: the end point of its last segment is
required to equal the start point of its first segment.  Contours carry no
orientation flag -- the signed-winding computation derives orientation
from the order of the coordinates themselves, so reversing every segment
of a contour reverses its winding contribution (this is exercised in the
tests and the demo).
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import List

from .errors import InvalidInputError

# Protocol limits (hard interface boundaries, see README).
MAX_CONTOURS = 20
MAX_CANVAS = 64

# Coordinates are accepted within this range.  The 16:1 guard band gives
# the rasterizer room to hold curves whose bbox touches the canvas edge;
# the value is part of the contract -- values outside are rejected, not
# clamped.
MAX_COORD_ABS = 16 * MAX_CANVAS


@dataclass(frozen=True)
class Point:
    """An immutable 2-D point.

    Coordinates given as ``int``/``float`` are stored as :class:`float`;
    :class:`fractions.Fraction` coordinates are stored exactly so the
    rational area entry point (:mod:`winding_fill.area`) can compute with
    exact rational arithmetic.
    """

    x: float
    y: float


@dataclass(frozen=True)
class Line:
    """Straight segment from ``p0`` to ``p1``."""

    p0: Point
    p1: Point


@dataclass(frozen=True)
class Quad:
    """Quadratic Bézier segment ``p0 -- control -- p1``."""

    p0: Point
    c: Point
    p1: Point


Segment = Line | Quad
"""Union alias for the two supported segment kinds."""


@dataclass(frozen=True)
class Contour:
    """A closed loop of :class:`Line`/:class:`Quad` segments."""

    segments: tuple[Segment, ...]


# ---------------------------------------------------------------------------
# Numeric validation
# ---------------------------------------------------------------------------


def _coordinate(v: object, where: str) -> float:
    """Validate one coordinate and return it as a plain float."""
    if isinstance(v, numbers.Rational) and not isinstance(v, bool):
        if isinstance(v, int):
            f = float(v)
            if abs(f) > MAX_COORD_ABS:
                raise InvalidInputError(
                    where, f"coordinate {f!r} exceeds limit |x| <= {MAX_COORD_ABS}"
                )
            return f
        # fractions.Fraction: keep the exact rational value.
        if abs(v) > MAX_COORD_ABS:
            raise InvalidInputError(
                where, f"coordinate {v!r} exceeds limit |x| <= {MAX_COORD_ABS}"
            )
        return v
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise InvalidInputError(
            where, f"expected a real number, got {type(v).__name__}"
        )
    f = float(v)
    if not math.isfinite(f):
        raise InvalidInputError(where, "NaN and Infinity are not accepted")
    if abs(f) > MAX_COORD_ABS:
        raise InvalidInputError(
            where, f"coordinate {f!r} exceeds limit |x| <= {MAX_COORD_ABS}"
        )
    return f


def _point(d: object, where: str) -> Point:
    if not isinstance(d, dict) or not ("x" in d and "y" in d):
        raise InvalidInputError(
            where, "point must be an object with numeric 'x' and 'y'"
        )
    x = _coordinate(d["x"], f"{where}.x")
    y = _coordinate(d["y"], f"{where}.y")
    return Point(x, y)


def canvas_dims(width: object, height: object) -> tuple[int, int]:
    """Validate raster grid dimensions.

    Integers only (``bool`` rejected), in ``1..MAX_CANVAS`` on each axis.
    Raises :class:`InvalidInputError` with a locatable message otherwise.
    """
    for name, v in (("width", width), ("height", height)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise InvalidInputError(
                name, f"expected an integer 1..{MAX_CANVAS}, got "
                f"{type(v).__name__}"
            )
        if not 1 <= v <= MAX_CANVAS:
            raise InvalidInputError(
                name, f"value {v} outside allowed range 1..{MAX_CANVAS}"
            )
    return int(width), int(height)


def parse_contours(raw: object) -> List[Contour]:
    """Validate raw nested data (dicts/tuples) into geometry objects.

    Accepted shape::

        [                          # <= MAX_CONTOURS contours
          [                        # one contour, list of segments
            {"type": "line", "p0": {"x":..,"y":..}, "p1": {...}},
            {"type": "quad", "p0": ..., "c": ..., "p1": ...},
            ...
          ], ...
        ]

    Rules:

    * each contour needs >= 3 segments;
    * consecutive segments must share endpoints (``seg[i].p1 == seg[i-1].p0``);
    * the contour must be closed: last ``p1`` == first ``p0`` (exact
      equality on the supplied numbers -- use :func:`build_contour` if you
      have open vertex chains);
    * zero-length lines and degenerate quads with coincident endpoints are
      rejected (they have no well-defined direction), but quads whose
      control point coincides with an endpoint are allowed (they collapse
      to a line and are handled exactly).
    """
    if not isinstance(raw, (list, tuple)):
        raise InvalidInputError(
            "contours", f"expected a list of contours, got {type(raw).__name__}"
        )
    if len(raw) == 0:
        raise InvalidInputError("contours", "at least one contour is required")
    if len(raw) > MAX_CONTOURS:
        raise InvalidInputError(
            f"contours", f"too many contours: {len(raw)} > {MAX_CONTOURS}"
        )

    contours: List[Contour] = []
    for ci, raw_con in enumerate(raw):
        cwhere = f"contours[{ci}]"
        if not isinstance(raw_con, (list, tuple)):
            raise InvalidInputError(
                cwhere, "contour must be a list of segments"
            )
        if len(raw_con) < 3:
            raise InvalidInputError(
                f"{cwhere}.segments",
                f"contour needs >= 3 segments, got {len(raw_con)}",
            )
        segs: list[Segment] = []
        for si, raw_seg in enumerate(raw_con):
            swhere = f"{cwhere}.segments[{si}]"
            if not isinstance(raw_seg, dict) or "type" not in raw_seg:
                raise InvalidInputError(
                    swhere, "segment must be an object with a 'type'"
                )
            kind = raw_seg["type"]
            if kind == "line":
                p0 = _point(raw_seg.get("p0"), f"{swhere}.p0")
                p1 = _point(raw_seg.get("p1"), f"{swhere}.p1")
                if p0 == p1:
                    raise InvalidInputError(
                        swhere, "zero-length line segment"
                    )
                segs.append(Line(p0, p1))
            elif kind == "quad":
                p0 = _point(raw_seg.get("p0"), f"{swhere}.p0")
                c = _point(raw_seg.get("c"), f"{swhere}.c")
                p1 = _point(raw_seg.get("p1"), f"{swhere}.p1")
                if p0 == p1:
                    raise InvalidInputError(
                        swhere,
                        "quadratic segment with coincident endpoints has no "
                        "well-defined direction",
                    )
                segs.append(Quad(p0, c, p1))
            else:
                raise InvalidInputError(
                    f"{swhere}.type",
                    f"unknown segment type {kind!r} (want 'line' or 'quad')",
                )
            # connectivity
            if si > 0:
                prev_end = _segment_end(segs[-2])
                cur_start = _segment_start(segs[-1])
                if prev_end != cur_start:
                    raise InvalidInputError(
                        swhere,
                        "segment start "
                        f"({cur_start.x}, {cur_start.y}) does not meet the "
                        f"previous segment end ({prev_end.x}, {prev_end.y})",
                    )
        if _segment_end(segs[-1]) != _segment_start(segs[0]):
            raise InvalidInputError(
                cwhere,
                "contour is not closed: last segment end != first segment "
                "start",
            )
        contours.append(Contour(tuple(segs)))
    return contours


def _segment_start(s: Segment) -> Point:
    return s.p0


def _segment_end(s: Segment) -> Point:
    return s.p1


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------


def build_contour(points: object, *, quads: object = ()) -> Contour:
    """Build a closed contour from an ordered vertex chain.

    ``points`` is a sequence of ``(x, y)`` pairs.  By default consecutive
    points are joined with lines and the chain is closed back to the first
    point.  ``quads`` optionally gives an iterable of ``(i, cx, cy)``
    triples: the edge from ``points[i]`` to ``points[(i+1) % n]`` becomes a
    quadratic Bézier with control point ``(cx, cy)``.

    At least 3 points are required.  Values undergo the same validation as
    :func:`parse_contours`.
    """
    if not isinstance(points, (list, tuple)):
        raise InvalidInputError(
            "points", f"expected a list of (x, y) pairs, got {type(points).__name__}"
        )
    n = len(points)
    if n < 3:
        raise InvalidInputError("points", f"need >= 3 points, got {n}")
    for pi, pt in enumerate(points):
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise InvalidInputError(
                f"points[{pi}]",
                "each point must be an (x, y) pair of real numbers",
            )
    if not isinstance(quads, (list, tuple)):
        raise InvalidInputError(
            "quads", "expected a list of (edge_index, cx, cy) triples"
        )
    quad_map: dict[int, tuple[float, float]] = {}
    for item in quads:
        if (
            not isinstance(item, tuple)
            or len(item) != 3
            or isinstance(item[0], bool)
            or not isinstance(item[0], int)
        ):
            raise InvalidInputError(
                "quads", "each entry must be (edge_index:int, cx, cy)"
            )
        i = item[0]
        if not 0 <= i < n:
            raise InvalidInputError(
                f"quads[{i}]", f"edge index {i} out of range 0..{n - 1}"
            )
        if i in quad_map:
            raise InvalidInputError(f"quads[{i}]", "duplicate edge index")
        quad_map[i] = (item[1], item[2])

    raw_segments: list[dict] = []
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        base = {
            "p0": {"x": x0, "y": y0},
            "p1": {"x": x1, "y": y1},
        }
        if i in quad_map:
            cx, cy = quad_map[i]
            base["type"] = "quad"
            base["c"] = {"x": cx, "y": cy}
        else:
            base["type"] = "line"
        raw_segments.append(base)
    return parse_contours([raw_segments])[0]
