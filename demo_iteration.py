"""Demo for the exact rational gray-level area entry point.

Computes, in well under 90 seconds and fully offline:

1. a normal scene -- a glyph-like polygon set (CCW outline, CW hole,
   overlapping same-direction ring) rendered as an exact-area gray map;
2. boundary results -- edges exactly on pixel boundaries (pure area, no
   line width), a slanted triangle with hand-checkable half pixels, and
   a window spanning negative coordinates;
3. an actually triggered rejection -- quadratic Bezier segments are
   outside the exact-rational polygon contract.

No sleep, no prerecorded output: every number printed is computed here.
"""

from __future__ import annotations

from fractions import Fraction

from winding_fill import (
    InvalidInputError,
    build_contour,
    fill_area,
)

_SHADES = " .:-=+*#%@"  # 10 gray levels, dark -> bright


def print_gray_map(res, title):
    print(f"--- {title} ---")
    print(
        f"window: left={res.left} bottom={res.bottom} "
        f"{res.width}x{res.height} (row 0 = bottom, printed top first)"
    )
    for row in reversed(res.areas):  # print top row first
        line = "".join(
            _SHADES[min(9, int(area * 10))] for area in row
        )
        fracs = " ".join(f"{area!s:>4}" for area in row)
        print(f"|{line}|  {fracs}")
    print(f"total covered area: {res.total_area()} (exact)")


def main():
    # 1. Normal scene: outline + reversed hole + overlapping ring.
    outer = build_contour([(0, 0), (7, 0), (7, 6), (0, 6)])
    hole = build_contour([(2, 2), (2, 4), (4, 4), (4, 2)])  # CW: hole
    ring = build_contour([(5, 1), (9, 1), (9, 4), (5, 4)])  # CCW overlap
    res = fill_area([outer, hole, ring], 0, 0, 10, 6)
    print_gray_map(res, "normal: outline + hole + overlapping ring")
    assert res.area_at(1, 1) == 1
    assert res.area_at(3, 3) == 0          # inside the hole
    assert res.area_at(6, 2) == 1          # overlap counts once
    print()

    # 2a. Boundary: edges exactly on pixel boundaries -> no line width.
    square = build_contour([(0, 0), (3, 0), (3, 3), (0, 3)])
    res_sq = fill_area([square], -1, -1, 5, 5)
    print_gray_map(res_sq, "boundary: axis-aligned square on pixel grid")
    assert res_sq.area_at(-1, 0) == 0 and res_sq.area_at(3, 1) == 0
    assert res_sq.total_area() == 9
    print()

    # 2b. Boundary: slanted triangle, hand-checkable halves.
    tri = build_contour([(0, 0), (3, 0), (0, 3)])
    res_tri = fill_area([tri], 0, 0, 4, 4)
    print_gray_map(res_tri, "boundary: slanted triangle (0,0)-(3,0)-(0,3)")
    assert res_tri.area_at(2, 0) == Fraction(1, 2)
    assert res_tri.total_area() == Fraction(9, 2)
    print("hand check: pixel (2,0) = 1/2, total = 9/2 = triangle area")
    print()

    # 2c. Boundary: window spanning negative coordinates.
    tri_neg = build_contour([(-2, -2), (1, -2), (-2, 1)])
    res_neg = fill_area([tri_neg], -3, -3, 5, 5)
    print_gray_map(res_neg, "boundary: triangle across negative coordinates")
    assert res_neg.area_at(-1, -1) == Fraction(1, 2)
    assert res_neg.total_area() == Fraction(9, 2)
    print()

    # 3. Actually triggered rejection: quad segment in the area entry.
    curve = build_contour([(0, 0), (2, 0), (2, 2)], quads=((1, 3, 1),))
    try:
        fill_area([curve], 0, 0, 2, 2)
    except InvalidInputError as exc:
        print(f"--- rejection: quadratic segment in exact area fill ---")
        print(f"InvalidInputError: {exc}")
    else:  # pragma: no cover - demo guard
        raise AssertionError("expected InvalidInputError")


if __name__ == "__main__":
    main()
