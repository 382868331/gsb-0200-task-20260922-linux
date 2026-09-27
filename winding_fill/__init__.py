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

from .core import (
    FillRule,
    SUBSAMPLES,
    CoverageResult,
    rasterize,
    rasterize_raw,
)
from .errors import InvalidInputError, WindingFillError
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
    "CoverageResult",
    "Contour",
    "FillRule",
    "InvalidInputError",
    "Line",
    "Point",
    "Quad",
    "WindingFillError",
    "build_contour",
    "parse_contours",
    "rasterize",
    "rasterize_raw",
]

__version__ = "1.0.0"
