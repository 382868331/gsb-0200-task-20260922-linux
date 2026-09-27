"""winding_fill -- nonzero-winding polygon/Bezier fill rasterizer.

Reusable offline library for an offline font-rasterization kernel:
closed contours made of line segments and quadratic Beziers are
rasterized using a fixed 4x4 sub-sample grid per pixel.  The fill
rule is selectable: nonzero winding (default, ``rule="nonzero"``)
or alternating even-odd parity (``rule="even-odd"``); both rules
share the exact same intersection computation.

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
    result.coverage_at(3, 3)   # 1.0 (nonzero rule)
    result.winding_at(3, 3, 0, 0)  # +1

    parity = rasterize_raw(raw_data, 8, 8, rule="even-odd")

Coordinate space: x right, y up; pixel (px, py) covers
``[px, px+1] x [py, py+1]`` and grid row 0 is the bottom row.
"""

from __future__ import annotations

from .core import (
    EVEN_ODD_RULE,
    FILL_RULES,
    NONZERO_RULE,
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
    "EVEN_ODD_RULE",
    "FILL_RULES",
    "InvalidInputError",
    "Line",
    "Point",
    "Quad",
    "NONZERO_RULE",
    "WindingFillError",
    "build_contour",
    "parse_contours",
    "rasterize",
    "rasterize_raw",
]

__version__ = "1.0.0"
