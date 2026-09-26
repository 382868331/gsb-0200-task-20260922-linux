#!/usr/bin/env python3
"""Fixed-input demo for the exact-area grayscale entry point.

Running ``python demo_iteration.py`` really rasterizes polygon contours
with exact rational per-pixel areas (nonzero winding), prints a gray
map for a glyph crossing negative coordinates, then shows boundary
results:

* an edge running exactly along pixel boundaries (no extra line width);
* a self-intersecting bowtie (1/2 areas on every touched pixel);
* the same contour wound twice (area counted once, not doubled).

It also triggers two real rejections: a quadratic Bezier segment fed to
the line-only exact-area rasterizer, and a ``bool`` pretending to be an
integer window origin.  No sleeps, no pre-recorded output, no third-
party dependencies.
"""

from __future__ import annotations

import time

from winding_fill import (
    InvalidInputError,
    build_contour,
    rasterize_area,
)

_GRAY = " .:-=+*#%@"


def _chain(points):
    """Line-only closed polygon from a vertex chain."""
    return build_contour(points)


def _shade(frac) -> str:
    return _GRAY[round(float(frac) * (len(_GRAY) - 1))]


def _print_map(result, title):
    print(f"--- {title} ---")
    print(f"window origin=({result.origin_x},{result.origin_y}) "
          f"size={result.width}x{result.height}; y axis points up")
    for py in range(result.height - 1, -1, -1):
        print("  " + "".join(_shade(result.area_at(px, py))
                              for px in range(result.width)))
    print(f"exact total covered area in window: {result.total_area()}")
    print()


def main() -> None:
    t_start = time.perf_counter()

    # Normal glyph: CCW slanted outer polygon, a CCW overlapping ring and
    # a CW triangular hole. Window starts at (-1,-1).
    outer = _chain([(0, 0), (10, 1), (11, 9), (1, 8)])
    overlap = _chain([(7, 3), (12, 3), (12, 8), (7, 8)])
    hole = _chain([(3, 2), (6, 6), (6, 2)])  # clockwise triangle

    glyph = rasterize_area(
        [outer, overlap, hole], 14, 11, origin=(-1, -1)
    )
    # The overlap pixel has winding +2 but must be counted once: pick a
    # pixel strictly inside the shared overlap and show value 1.
    _print_map(glyph, "normal result: slanted polygon + overlap + CW hole")
    overlap_px = glyph.area_at(9, 5)  # local coords inside window
    print(f"overlap pixel (global (8,4)) winding is +2 but value = "
          f"{overlap_px} (not 2)\n")

    # Boundary 1: edge exactly on pixel boundaries -- exact 0/1 values.
    grid_square = rasterize_area(
        [_chain([(0, 0), (2, 0), (2, 2), (0, 2)])], 3, 3
    )
    _print_map(grid_square, "boundary: edges exactly on pixel boundaries")

    # Boundary 2: self-intersecting bowtie, every touched pixel is 1/2.
    bowtie = rasterize_area(
        [_chain([(0, 0), (2, 2), (2, 0), (0, 2)])], 2, 2
    )
    _print_map(bowtie, "boundary: self-intersecting bowtie")

    # Boundary 3: same contour wound twice -- no doubled brightness.
    sq = _chain([(0, 0), (2, 0), (2, 2), (0, 2)])
    doubled = rasterize_area([sq, sq], 2, 2)
    print("--- boundary: same square wound twice ---")
    print(f"pixel (0,0) value = {doubled.area_at(0, 0)}; "
          f"total = {doubled.total_area()} (not 8)\n")

    # Real rejection 1: quadratic curves are not accepted here.
    curved = build_contour([(0, 0), (2, 0), (2, 2)], quads=((0, 1, -1),))
    try:
        rasterize_area([curved], 4, 4)
    except InvalidInputError as exc:
        print(f"rejected curved input -> {exc}\n")

    # Real rejection 2: bool is not an integer origin.
    try:
        rasterize_area([sq], 2, 2, origin=(True, 0))
    except InvalidInputError as exc:
        print(f"rejected bool origin -> {exc}\n")

    elapsed = time.perf_counter() - t_start
    print(f"all exact computations finished in {elapsed:.3f} s "
          f"(budget 90 s)")


if __name__ == "__main__":
    main()
