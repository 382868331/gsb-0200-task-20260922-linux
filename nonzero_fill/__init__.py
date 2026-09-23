"""nonzero_fill：多边形字体轮廓的非零环绕填充（4x4 子采样）。

快速上手::

    from nonzero_fill import point, line, quad, outline, rasterize

    glyph = outline([
        [line(point(0, 0), point(4, 0)),
         line(point(4, 0), point(4, 4)),
         line(point(4, 4), point(0, 4)),
         line(point(0, 4), point(0, 0))],
    ])
    result = rasterize(glyph, 8, 8)
    result.coverage_at(1, 1)     # 1.0
    result.winding_at(1, 1, 0, 0)  # 1

详见 README.md。
"""

from __future__ import annotations

from .errors import (
    InvalidCanvasError,
    InvalidContourError,
    InvalidCoordinateError,
    InvalidQueryError,
    NonzeroFillError,
)
from .geometry import LineSegment, MonoPiece, QuadraticSegment
from .model import MAX_CONTOURS, GlyphOutline, line, outline, point, quad
from .rasterizer import MAX_CANVAS, SUBSAMPLES, RasterResult, rasterize

__all__ = [
    "point",
    "line",
    "quad",
    "outline",
    "rasterize",
    "GlyphOutline",
    "LineSegment",
    "QuadraticSegment",
    "MonoPiece",
    "RasterResult",
    "MAX_CONTOURS",
    "MAX_CANVAS",
    "SUBSAMPLES",
    "NonzeroFillError",
    "InvalidCoordinateError",
    "InvalidContourError",
    "InvalidCanvasError",
    "InvalidQueryError",
]
