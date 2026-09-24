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
    InvalidInputError,
    WindingFillError,
    boolean_difference,
    boolean_intersection,
    boolean_union,
    build_contour,
    rasterize,
    rasterize_raw,
    signed_area,
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

    t0 = time.perf_counter()
    result = rasterize([outer, overlap, hole_rev], width, height)
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


def boolean_demo() -> None:
    print("=== boolean ops on two simple rational polygons ===")
    # Two CCW unit-grid squares overlapping in a 2x2 block; every area
    # below is computed exactly (fractions.Fraction inside the library).
    a = build_contour([(0, 0), (4, 0), (4, 4), (0, 4)])
    b = build_contour([(2, 2), (6, 2), (6, 6), (2, 6)])

    t0 = time.perf_counter()
    u = boolean_union(a, b)
    inter = boolean_intersection(a, b)
    diff = boolean_difference(a, b)
    elapsed = time.perf_counter() - t0

    print("  A = [0,4]x[0,4]  (area 16), B = [2,6]x[2,6] (area 16)")
    print(f"  union area        = {u.area}  (hand value 28 = 16+16-4)")
    print(f"  intersection area = {inter.area}  (hand value 4)")
    print(f"  difference A\\B    = {diff.area}  (hand value 12 = 16-4)")
    print(f"  computed in {elapsed * 1000:.2f} ms\n")

    # Rasterize the union polygon with the existing nonzero-winding
    # rasterizer (4x4 sub-samples).
    width, height = 8, 8
    result = rasterize(list(u.contours), width, height)
    print(f"  rasterized union on a {width}x{height} canvas (y up):")
    print("    " + "".join(str(x % 10) for x in range(width)))
    for py in range(height - 1, -1, -1):
        print(f"{py:2d}  " + "".join(
            _shade(result.levels[py][px]) for px in range(width)
        ))
    print("  A-only pixel (1,1) solid:", result.coverage_at(1, 1) == 1.0,
          "| B-only pixel (5,5) solid:",
          result.coverage_at(5, 5) == 1.0,
          "| outside empty:", result.coverage_at(7, 7) == 0.0)
    assert result.coverage_at(1, 1) == 1.0
    assert result.coverage_at(5, 5) == 1.0
    assert result.coverage_at(7, 7) == 0.0
    print()

    print("  -- boundary relations --")
    # Containment: outer minus inner is a ring reported as CCW outer + CW
    # hole loop; the hole pixel rasterizes empty.
    outer = build_contour([(0, 0), (8, 0), (8, 8), (0, 8)])
    inner = build_contour([(2, 2), (6, 2), (6, 6), (2, 6)])
    ring = boolean_difference(outer, inner)
    ring_raster = rasterize(list(ring.contours), 10, 10)
    print(
        f"  containment A\\B: area {ring.area} (=64-16), "
        f"{len(ring.contours)} loops "
        f"[outer {signed_area(ring.contours[0])}, "
        f"hole {signed_area(ring.contours[1])}], "
        f"hole center covered={ring_raster.coverage_at(4, 4)}"
    )
    assert ring.area == 48
    assert ring_raster.coverage_at(4, 4) == 0.0

    # Pure edge contact: B shares a segment of A's right edge, so the
    # intersection has zero area -> empty result.
    tangent = build_contour([(4, 1), (6, 1), (6, 3), (4, 3)])
    t_inter = boolean_intersection(a, tangent)
    t_union = boolean_union(a, tangent)
    print(
        f"  edge-touching squares: intersection area {t_inter.area} "
        f"(0 loops), union area {t_union.area} (=16+4)"
    )
    assert t_inter.area == 0 and t_inter.contours == ()
    assert t_union.area == 20
    print()


def boolean_rejections() -> None:
    print("=== boolean input rejection (really triggered) ===")
    other = build_contour([(0, 0), (6, 0), (6, 6), (0, 6)])

    bowtie = build_contour([(1, 1), (5, 5), (5, 1), (1, 5)])
    try:
        boolean_union(bowtie, other)
    except InvalidInputError as exc:
        print(f"  rejected self-crossing contour -> {type(exc).__name__}")
        print(f"    location -> {exc.location}: {exc.message}")
    else:  # pragma: no cover - defensive
        raise AssertionError("self-intersecting contour was not rejected")

    curved = build_contour(
        [(0, 0), (4, 0), (4, 4), (0, 4)], quads=((0, 2, -1),)
    )
    try:
        boolean_intersection(curved, other)
    except InvalidInputError as exc:
        print(f"  rejected quadratic contour     -> {type(exc).__name__}")
        print(f"    location -> {exc.location}")
    else:  # pragma: no cover - defensive
        raise AssertionError("curved contour was not rejected")
    print()


def main() -> None:
    normal_result()
    boolean_demo()
    boolean_rejections()
    rejection_boundary()
    print("demo finished successfully.")


if __name__ == "__main__":
    main()
