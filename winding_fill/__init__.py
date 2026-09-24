"""winding_fill -- nonzero-winding polygon/Bezier fill rasterizer.

Reusable offline library for an offline font-rasterization kernel:
closed contours made of line segments and quadratic Beziers are
rasterized with the nonzero winding rule using a fixed 4x4
sub-sample grid per pixel.

Quick start::

    from winding_fill import rasterize_raw

    result = rasterize_raw(
        [
            [  # one contour: CCW square
                {"type": "line", "p0": {"x": 1, "y": 1}, "p1": {"x": 5, "y": 1}},
                {"type": "line", "p0": {"x": 5, "y": 1}, "p1": {"x": 5, "y": 5}},
                {"type": "line", "p0": {"x": 5, "y": 5}, "p1": {"x": 1, "y": 5}},
                {"type": "line", "p0": {"x": 1, "y": 5}, "p1": {"x": 1, "y": 1}},
            ]
        ],
        width=8,
        height=8,
    )
    result.coverage_at(3, 3)   # 1.0
    result.winding_at(3, 3, 0, 0)  # +1

Coordinate space: x right, y up; pixel (px, py) covers
``[px, px+1] x [py, py+1]`` and grid row 0 is the bottom row.
"""

from __future__ import annotations

from .boolean import (
    DIFFERENCE,
    INTERSECTION,
    UNION,
    BooleanResult,
    boolean,
    boolean_raw,
    difference,
    intersection,
    polygon_area,
    union,
)
from .core import (
    SUBSAMPLES,
    CoverageResult,
    rasterize,
    rasterize_raw,
)
from .errors import (
    InvalidInputError,
    SelfIntersectionError,
    WindingFillError,
)
from .geometry import (
    MAX_CANVAS,
    MAX_CONTOURS,
    MAX_COORD_ABS,
    Contour,
    Line,
    Point,
    Quad,
    build_contour,
    parse_contours,
)

__all__ = [
    "SUBSAMPLES",
    "MAX_CANVAS",
    "MAX_CONTOURS",
    "MAX_COORD_ABS",
    "BooleanResult",
    "Contour",
    "DIFFERENCE",
    "INTERSECTION",
    "InvalidInputError",
    "Line",
    "Point",
    "Quad",
    "SelfIntersectionError",
    "UNION",
    "WindingFillError",
    "boolean",
    "boolean_raw",
    "build_contour",
    "difference",
    "intersection",
    "parse_contours",
    "polygon_area",
    "rasterize",
    "rasterize_raw",
    "union",
]

__version__ = "1.1.0"
