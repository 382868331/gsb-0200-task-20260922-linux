"""Tests for the exact rational gray-level area entry point.

Hand-computed cases (slanted triangle, reversed hole, overlapping
same-direction rings, negative coordinates, edges on pixel boundaries,
self-intersection, double winding) plus a sum-consistency check of all
pixel areas against an independent whole-window clipped-area reference
(tests/independent_reference.py, which does not import the core).
"""

from __future__ import annotations

import unittest
from fractions import Fraction

from winding_fill import (
    AreaResult,
    InvalidInputError,
    build_contour,
    fill_area,
    fill_area_raw,
)

from independent_reference import reference_clipped_area


def _rings(contours):
    """Extract vertex rings from Contour objects for the reference."""
    rings = []
    for con in contours:
        rings.append([(seg.p0.x, seg.p0.y) for seg in con.segments])
    return rings


class TestSlantedTriangle(unittest.TestCase):
    """Triangle (0,0)-(3,0)-(0,3): x + y <= 3 in the first quadrant."""

    def setUp(self):
        self.tri = build_contour([(0, 0), (3, 0), (0, 3)])
        self.res = fill_area([self.tri], 0, 0, 4, 4)

    def test_hand_computed_pixels(self):
        half = Fraction(1, 2)
        expected = [
            [1, 1, half, 0],   # y = 0 row (bottom)
            [1, half, 0, 0],   # y = 1
            [half, 0, 0, 0],   # y = 2
            [0, 0, 0, 0],      # y = 3
        ]
        for py in range(4):
            for px in range(4):
                self.assertEqual(
                    self.res.area_at(px, py),
                    Fraction(expected[py][px]),
                    f"pixel ({px}, {py})",
                )

    def test_total_is_triangle_area(self):
        self.assertEqual(self.res.total_area(), Fraction(9, 2))

    def test_gray_at_float_view(self):
        self.assertEqual(self.res.gray_at(2, 0), 0.5)
        self.assertEqual(self.res.gray_at(0, 0), 1.0)


class TestReversedHole(unittest.TestCase):
    """CCW square [0,4]^2 with CW square hole [1,3]^2."""

    def setUp(self):
        outer = build_contour([(0, 0), (4, 0), (4, 4), (0, 4)])
        inner = build_contour([(1, 1), (1, 3), (3, 3), (3, 1)])
        self.res = fill_area([outer, inner], 0, 0, 4, 4)

    def test_ring_and_hole(self):
        for py in range(4):
            for px in range(4):
                inside_hole = 1 <= px < 3 and 1 <= py < 3
                self.assertEqual(
                    self.res.area_at(px, py),
                    Fraction(0 if inside_hole else 1),
                    f"pixel ({px}, {py})",
                )

    def test_total(self):
        self.assertEqual(self.res.total_area(), Fraction(12))


class TestOverlappingSameDirection(unittest.TestCase):
    """Two overlapping CCW squares: overlap counts once, not twice."""

    def test_overlap_counted_once(self):
        s1 = build_contour([(0, 0), (2, 0), (2, 2), (0, 2)])
        s2 = build_contour([(1, 1), (3, 1), (3, 3), (1, 3)])
        res = fill_area([s1, s2], 0, 0, 4, 4)
        # Union is an L shape of area 4 + 4 - 1 = 7.
        self.assertEqual(res.total_area(), Fraction(7))
        # The shared pixel is fully covered exactly once.
        self.assertEqual(res.area_at(1, 1), Fraction(1))
        self.assertEqual(res.area_at(2, 0), Fraction(0))
        self.assertEqual(res.area_at(0, 2), Fraction(0))


class TestNegativeCoordinates(unittest.TestCase):
    """Triangle (-2,-2)-(1,-2)-(-2,1): x >= -2, y >= -2, x+y <= -1."""

    def test_hand_computed(self):
        tri = build_contour([(-2, -2), (1, -2), (-2, 1)])
        res = fill_area([tri], -3, -3, 5, 5)
        half = Fraction(1, 2)
        self.assertEqual(res.area_at(-2, -2), Fraction(1))
        self.assertEqual(res.area_at(-1, -2), Fraction(1))
        self.assertEqual(res.area_at(-2, -1), Fraction(1))
        self.assertEqual(res.area_at(0, -2), half)
        self.assertEqual(res.area_at(-1, -1), half)
        self.assertEqual(res.area_at(-2, 0), half)
        self.assertEqual(res.area_at(0, 0), Fraction(0))
        self.assertEqual(res.total_area(), Fraction(9, 2))


class TestEdgeOnPixelBoundary(unittest.TestCase):
    """Edges exactly on pixel boundaries: pure area, no line width."""

    def test_axis_aligned_square(self):
        sq = build_contour([(0, 0), (2, 0), (2, 2), (0, 2)])
        res = fill_area([sq], -2, -2, 6, 6)
        for py in range(-2, 4):
            for px in range(-2, 4):
                inside = 0 <= px < 2 and 0 <= py < 2
                self.assertEqual(
                    res.area_at(px, py),
                    Fraction(1 if inside else 0),
                    f"pixel ({px}, {py})",
                )
        self.assertEqual(res.total_area(), Fraction(4))

    def test_diagonal_through_grid_points(self):
        # Hypotenuse passes exactly through pixel corners (1,1), (2,2).
        tri = build_contour([(0, 0), (3, 0), (0, 3)])
        res = fill_area([tri], 0, 0, 3, 3)
        self.assertEqual(res.area_at(1, 1), Fraction(1, 2))
        self.assertEqual(res.area_at(2, 2), Fraction(0))


class TestSelfIntersection(unittest.TestCase):
    def test_bowtie(self):
        # Both lobes have nonzero winding (+1 and -1): both filled.
        bow = build_contour([(0, 0), (2, 2), (2, 0), (0, 2)])
        res = fill_area([bow], 0, 0, 3, 3)
        self.assertEqual(res.total_area(), Fraction(2))
        self.assertEqual(res.area_at(0, 0), Fraction(1, 2))
        self.assertEqual(res.area_at(0, 1), Fraction(1, 2))
        self.assertEqual(res.area_at(1, 0), Fraction(1, 2))
        self.assertEqual(res.area_at(1, 1), Fraction(1, 2))

    def test_double_wound_square_counts_once(self):
        # One contour tracing the same square twice: winding is 2
        # inside, but the nonzero rule fills the square exactly once.
        sq = build_contour(
            [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0), (2, 0), (2, 2), (0, 2)]
        )
        res = fill_area([sq], 0, 0, 3, 3)
        self.assertEqual(res.area_at(0, 0), Fraction(1))
        self.assertEqual(res.area_at(1, 1), Fraction(1))
        self.assertEqual(res.total_area(), Fraction(4))


class TestFractionCoordinates(unittest.TestCase):
    def test_exact_fractional_geometry(self):
        tri = build_contour(
            [(Fraction(0), Fraction(0)),
             (Fraction(7, 2), Fraction(0)),
             (Fraction(0), Fraction(5, 3))]
        )
        res = fill_area([tri], 0, 0, 4, 2)
        self.assertEqual(
            res.total_area(), Fraction(7, 2) * Fraction(5, 3) / 2
        )
        # Pixel (3, 0): the hypotenuse cuts it; covered part is the
        # triangle with vertices (3,0), (7/2,0), (3, 5/3*(7/2-3)/ (7/2))
        # = (3, 5/21): area = 1/2 * 1/2 * 5/21 = 5/84.
        self.assertEqual(res.area_at(3, 0), Fraction(5, 84))


class TestSumConsistency(unittest.TestCase):
    """Sum of pixel areas == independent clipped nonzero area."""

    def _check(self, contours, left, bottom, width, height):
        res = fill_area(contours, left, bottom, width, height)
        expected = reference_clipped_area(
            _rings(contours), left, bottom, width, height
        )
        self.assertEqual(res.total_area(), expected)

    def test_triangle_window_clips_geometry(self):
        tri = build_contour([(0, 0), (5, 1), (1, 4)])
        for window in [(0, 0, 6, 5), (1, 0, 3, 2), (2, 1, 2, 2), (-1, -1, 8, 7)]:
            self._check([tri], *window)

    def test_hole_and_overlap(self):
        outer = build_contour([(0, 0), (6, 0), (6, 5), (0, 5)])
        hole = build_contour([(1, 1), (1, 3), (4, 3), (4, 1)])
        extra = build_contour([(3, 2), (7, 2), (7, 6), (3, 6)])
        for window in [(0, 0, 7, 6), (2, 2, 3, 3), (-2, -3, 10, 10)]:
            self._check([outer, hole, extra], *window)

    def test_self_intersecting_starish(self):
        # 4-point chain closed back to the start: a self-intersecting quad
        star = build_contour([(0, 0), (4, 0), (0, 3), (4, 3)])
        for window in [(0, 0, 5, 4), (1, 0, 3, 3)]:
            self._check([star], *window)

    def test_slanted_fractional_edges(self):
        tri = build_contour(
            [(Fraction(-3, 2), Fraction(-1)),
             (Fraction(9, 4), Fraction(-1)),
             (Fraction(0), Fraction(7, 3))]
        )
        for window in [(-2, -2, 5, 5), (0, 0, 3, 3), (-2, 1, 4, 2)]:
            self._check([tri], *window)


class TestAreaApi(unittest.TestCase):
    def setUp(self):
        self.sq = build_contour([(0, 0), (2, 0), (2, 2), (0, 2)])

    def test_result_shape_and_index_base(self):
        res = fill_area([self.sq], -1, -1, 4, 3)
        self.assertIsInstance(res, AreaResult)
        self.assertEqual((res.left, res.bottom), (-1, -1))
        self.assertEqual((res.width, res.height), (4, 3))
        self.assertEqual(len(res.areas), 3)
        self.assertEqual(len(res.areas[0]), 4)
        self.assertEqual(res.areas[0][0], res.area_at(-1, -1))

    def test_fill_area_raw(self):
        raw = [[
            {"type": "line", "p0": {"x": 0, "y": 0}, "p1": {"x": 2, "y": 0}},
            {"type": "line", "p0": {"x": 2, "y": 0}, "p1": {"x": 2, "y": 2}},
            {"type": "line", "p0": {"x": 2, "y": 2}, "p1": {"x": 0, "y": 2}},
            {"type": "line", "p0": {"x": 0, "y": 2}, "p1": {"x": 0, "y": 0}},
        ]]
        res = fill_area_raw(raw, 0, 0, 3, 3)
        self.assertEqual(res.area_at(0, 0), Fraction(1))
        self.assertEqual(res.total_area(), Fraction(4))

    def test_area_at_out_of_range(self):
        res = fill_area([self.sq], 0, 0, 3, 3)
        with self.assertRaises(InvalidInputError):
            res.area_at(3, 0)
        with self.assertRaises(InvalidInputError):
            res.area_at(0, -1)
        with self.assertRaises(InvalidInputError):
            res.area_at(0.5, 0)

    def test_rejects_quad_segments(self):
        curve = build_contour([(0, 0), (2, 0), (2, 2)], quads=((1, 3, 1),))
        with self.assertRaises(InvalidInputError) as cm:
            fill_area([curve], 0, 0, 2, 2)
        self.assertIn("segments[1]", str(cm.exception))

    def test_rejects_bad_window(self):
        for bad in [
            (True, 0, 2, 2),          # bool origin
            (0, 0, True, 2),          # bool size
            (0, 0, 0, 2),             # zero width
            (0.5, 0, 2, 2),           # non-integer origin
            (0, 0, 65, 2),            # size beyond MAX_CANVAS
            (-2000, 0, 2, 2),         # origin beyond coordinate limit
        ]:
            with self.assertRaises(InvalidInputError, msg=f"window {bad}"):
                fill_area([self.sq], *bad)

    def test_rejects_bad_contours(self):
        with self.assertRaises(InvalidInputError):
            fill_area([], 0, 0, 2, 2)
        with self.assertRaises(InvalidInputError):
            fill_area([self.sq, "nope"], 0, 0, 2, 2)
        with self.assertRaises(InvalidInputError):
            fill_area(self.sq, 0, 0, 2, 2)  # not a list of contours

    def test_old_subsample_api_still_works(self):
        from winding_fill import rasterize

        res = rasterize([self.sq], 4, 4)
        self.assertEqual(res.coverage_at(0, 0), 1.0)
        self.assertEqual(res.coverage_at(3, 3), 0.0)


if __name__ == "__main__":
    unittest.main()
