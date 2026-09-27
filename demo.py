#!/usr/bin/env python3
"""Fixed-input demo for the winding_fill library.

Running ``python demo.py`` really rasterizes a small glyph and prints its
gray-level map plus a few sub-sample winding queries, then demonstrates
one input-rejection boundary actually being triggered.  There are no
sleeps, no pre-recorded outputs and no third-party dependencies.

The glyph (coordinates are integers/halves, x right, y up):

    * an outer CCW rectangle;
    * a second CCW rectangle overlapping its right side (their overlap
      has winding number +2 under nonzero winding);
    * a clockwise contour with two curved edges cutting a hole (the
      winding there returns to 0).
"""

from __future__ import annotations

import math
import time

from winding_fill import (
    FillRule,
    InvalidInputError,
    WindingFillError,
    build_contour,
    rasterize,
    rasterize_raw,
)

# 17 gray steps for coverage levels 0..16 (k/16).
_GRAY = " .:-=+*#%@"


def _shade(level: int) -> str:
    # level 0 -> space; level 16 -> '@'
    idx = round(level / 16 * (len(_GRAY) - 1))
    return _GRAY[idx]


def normal_result() -> None:
    width, height = 19, 13
    outer = build_contour(
        [(1, 1), (13, 1), (13, 11), (1, 11)],
        quads=((0, 7, -0.5), (2, 7, 12.5)),  # gently curved bottom/top
    )
    overlap = build_contour(
        [(9, 4), (17, 4), (17, 9), (9, 9)],
        quads=((0, 13, 2.5),),
    )
    # Clockwise (reversed) curved hole.
    hole = build_contour(
        [(4, 3), (9, 3), (9, 8), (4, 8)],
        quads=((0, 6.5, 1.5), (2, 6.5, 9.5)),
    )
    hole_rev = _reverse_contour(hole)

    contours = [outer, overlap, hole_rev]
    t0 = time.perf_counter()
    result = rasterize(contours, width, height)
    elapsed = time.perf_counter() - t0

    print(f"=== nonzero-winding fill: {width}x{height} canvas, 4x4 sub-samples ===")
    print("    " + "".join(str(x % 10) for x in range(width)))
    for py in range(height - 1, -1, -1):
        row = "".join(_shade(result.levels[py][px]) for px in range(width))
        print(f"{py:2d}  {row}")
    print("legend: ' '=0/16 ... '@'=16/16 covered\n")

    print(f"computed in {elapsed * 1000:.1f} ms\n")

    # Fixed query points.
    queries = [
        # px, py, i, j, description
        (2, 6, 1, 2, "inside outer rectangle only"),
        (11, 6, 1, 2, "outer + overlap (winding should be +2)"),
        (6, 5, 1, 2, "inside the curved hole (winding 0)"),
        (18, 11, 0, 0, "completely outside"),
    ]
    print("sub-sample winding queries:")
    for px, py, i, j, desc in queries:
        w = result.winding_at(px, py, i, j)
        cov = result.coverage_at(px, py)
        print(
            f"  pixel({px:2d},{py:2d}) sample[{i},{j}] "
            f"winding={w:+d}  pixel coverage={cov:5.3f}   # {desc}"
        )
    print()

    # Hard invariants shown on the real result.
    overlap_pixel = result.windings_at(11, 6)
    assert all(w == 2 for w in overlap_pixel), overlap_pixel
    assert result.coverage_at(6, 5) == 0.0
    assert result.coverage_at(2, 6) == 1.0

    # Same geometry with the alternative even-odd rule.  Crossing
    # computation is identical (same winding numbers); only the
    # coverage reduction changes.
    result_eo = rasterize(contours, width, height, rule=FillRule.EVEN_ODD)
    assert result_eo.windings == result.windings
    print("=== even-odd fill of the same glyph ===")
    print("    " + "".join(str(x % 10) for x in range(width)))
    for py in range(height - 1, -1, -1):
        row = "".join(_shade(result_eo.levels[py][px]) for px in range(width))
        print(f"{py:2d}  {row}")
    print("legend: ' '=0/16 ... '@'=16/16 covered\n")
    eo_queries = [
        (2, 6, 1, 2, "inside outer only (odd -> filled)"),
        (11, 6, 1, 2, "outer + overlap (2 crossings -> toggled OFF)"),
        (6, 5, 1, 2, "inside the reversed hole (0 crossings, empty under both)"),
    ]
    print("even-odd sub-sample queries:")
    for px, py, i, j, desc in eo_queries:
        print(
            f"  pixel({px:2d},{py:2d}) sample[{i},{j}] "
            f"parity={result_eo.parity_at(px, py, i, j):+d}  "
            f"winding={result_eo.winding_at(px, py, i, j):+d}   # {desc}"
        )
    assert result_eo.coverage_at(11, 6) == 0.0
    assert result_eo.coverage_at(6, 5) == 0.0
    assert result_eo.coverage_at(2, 6) == 1.0
    print()


def _reverse_contour(contour):
    from winding_fill import Line, Quad

    rev = []
    for seg in reversed(contour.segments):
        if isinstance(seg, Line):
            rev.append(Line(seg.p1, seg.p0))
        else:
            rev.append(Quad(seg.p1, seg.c, seg.p0))
    return type(contour)(tuple(rev))


def rejection_boundary() -> None:
    print("=== input rejection boundary (really triggered) ===")
    # A coordinate of NaN must be refused with a locatable error rather
    # than silently producing a corrupted canvas.
    bad_glyph = [
        [
            {"type": "line", "p0": {"x": 0, "y": 0},
             "p1": {"x": 4, "y": 0}},
            {"type": "line", "p0": {"x": 4, "y": 0},
             "p1": {"x": 4, "y": math.nan}},  # <-- bad value, located
            {"type": "line", "p0": {"x": 4, "y": math.nan},
             "p1": {"x": 0, "y": 0}},
        ]
    ]
    try:
        rasterize_raw(bad_glyph, 8, 8)
    except WindingFillError as exc:
        assert isinstance(exc, InvalidInputError)
        print(f"  rejected NaN coordinate -> {type(exc).__name__}: {exc}")
        print(f"  error location token   -> {exc.location!r}")
    else:  # pragma: no cover - defensive, must never happen
        raise AssertionError("NaN input was not rejected")

    # A bool masquerading as an integer canvas dimension is refused too.
    try:
        rasterize_raw(
            [[
                {"type": "line", "p0": {"x": 0, "y": 0},
                 "p1": {"x": 1, "y": 0}},
                {"type": "line", "p0": {"x": 1, "y": 0},
                 "p1": {"x": 1, "y": 1}},
                {"type": "line", "p0": {"x": 1, "y": 1},
                 "p1": {"x": 0, "y": 0}},
            ]],
            True, 8,
        )
    except InvalidInputError as exc:
        print(f"  rejected bool width    -> {type(exc).__name__}: {exc}")
    print()


def main() -> None:
    normal_result()
    rejection_boundary()
    print("demo finished successfully.")


if __name__ == "__main__":
    main()
