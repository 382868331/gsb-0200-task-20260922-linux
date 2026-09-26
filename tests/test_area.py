"""Tests for the exact rational grayscale area entry point.

Two kinds of checks:

* hand-computed pixel areas (slanted triangle, reverse hole,
  overlapping same-direction rings, negative coordinates, edges on
  pixel boundaries, zero-width slit, self-intersecting bowtie);
* exact agreement with the independent reference in
  :mod:`area_reference` (never imports :mod:`winding_fill`), both
  per-pixel and for the sum of all pixel areas vs. the clipped fill
  area of the whole window.
"""

from __future__ import annotations

import random
import unittest
from fractions import Fraction

from winding_fill import (
    AreaCoverageResult,
    InvalidInputError,
    build_contour,
    rasterize_area,
    rasterize_area_raw,
)

from area_reference import reference_pixel_area, reference_window_area


def _chain(points):
    """Raw closed line-only contour from a vertex chain."""
    raw = []
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        raw.append(
            {
                "type": "line",
                "p0": {"x": x0, "y": y0},
                "p1": {"x": x1, "y": y1},
            }
        )
    return raw


def _areas(result):
    return tuple(tuple(row) for row in result.areas)


class HandComputedTests(unittest.TestCase):
    def test_slanted_triangle(self):
        # Triangle (0,0)-(2,0)-(0,1); hypotenuse y = 1 - x/2.
        raw = [_chain([(0, 0), (2, 0), (0, 1)])]
        res = rasterize_area_raw(raw, 2, 1)
        self.assertEqual(res.area_at(0, 0), Fraction(3, 4))
        self.assertEqual(res.area_at(1, 0), Fraction(1, 4))
        self.assertEqual(res.total_area(), Fraction(1))

    def test_reverse_hole(self):
        # CCW outer square with a CW inner square hole.
        raw = [
            _chain([(0, 0), (4, 0), (4, 4), (0, 4)]),
            _chain([(1, 1), (1, 3), (3, 3), (3, 1)]),  # clockwise
        ]
        res = rasterize_area_raw(raw, 4, 4)
        hole = {(1, 1), (2, 1), (1, 2), (2, 2)}
        for py in range(4):
            for px in range(4):
                want = Fraction(0) if (px, py) in hole else Fraction(1)
                self.assertEqual(res.area_at(px, py), want, (px, py))
        self.assertEqual(res.total_area(), Fraction(12))

    def test_overlapping_same_direction_rings(self):
        # Two CCW squares overlapping on pixel (1,1): winding +2 there,
        # but the area must be counted once, not twice.
        raw = [
            _chain([(0, 0), (2, 0), (2, 2), (0, 2)]),
            _chain([(1, 1), (3, 1), (3, 3), (1, 3)]),
        ]
        res = rasterize_area_raw(raw, 3, 3)
        covered = {
            (0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2),
        }
        for py in range(3):
            for px in range(3):
                want = Fraction(1) if (px, py) in covered else Fraction(0)
                self.assertEqual(res.area_at(px, py), want, (px, py))
        self.assertEqual(res.total_area(), Fraction(7))

    def test_double_wound_region_not_doubled(self):
        # The same square wound twice: brightness stays 1, not 2.
        square = _chain([(0, 0), (2, 0), (2, 2), (0, 2)])
        res = rasterize_area_raw([square, square], 3, 3)
        for py in range(3):
            for px in range(3):
                want = Fraction(1) if px < 2 and py < 2 else Fraction(0)
                self.assertEqual(res.area_at(px, py), want, (px, py))
        self.assertEqual(res.total_area(), Fraction(4))

    def test_negative_coordinates(self):
        # Triangle (-2,-2)-(1,-2)-(-2,1); hypotenuse y = -x - 1.
        raw = [_chain([(-2, -2), (1, -2), (-2, 1)])]
        res = rasterize_area_raw(raw, 3, 3, origin=(-2, -2))
        expect = {
            (-2, -2): Fraction(1), (-1, -2): Fraction(1),
            (0, -2): Fraction(1, 2),
            (-2, -1): Fraction(1), (-1, -1): Fraction(1, 2),
            (0, -1): Fraction(0),
            (-2, 0): Fraction(1, 2), (-1, 0): Fraction(0),
            (0, 0): Fraction(0),
        }
        for (px, py), want in expect.items():
            self.assertEqual(res.area_at(px + 2, py + 2), want, (px, py))
        self.assertEqual(res.total_area(), Fraction(9, 2))

    def test_edge_exactly_on_pixel_boundary(self):
        # Square on integer grid: boundary edges add no extra width.
        raw = [_chain([(0, 0), (2, 0), (2, 2), (0, 2)])]
        res = rasterize_area_raw(raw, 3, 3)
        for py in range(3):
            for px in range(3):
                want = Fraction(1) if px < 2 and py < 2 else Fraction(0)
                self.assertEqual(res.area_at(px, py), want, (px, py))

    def test_zero_width_slit_has_no_area(self):
        # A slit out and back along x=1 adds no area anywhere.
        raw = [
            _chain(
                [(0, 0), (2, 0), (2, 2), (1, 2), (1, 1), (1, 2), (0, 2)]
            )
        ]
        res = rasterize_area_raw(raw, 2, 2)
        for py in range(2):
            for px in range(2):
                self.assertEqual(res.area_at(px, py), Fraction(1), (px, py))
        self.assertEqual(res.total_area(), Fraction(4))

    def test_self_intersecting_bowtie(self):
        # Bowtie (0,0)-(2,2)-(2,0)-(0,2): both lobes covered, each
        # pixel exactly half.
        raw = [_chain([(0, 0), (2, 2), (2, 0), (0, 2)])]
        res = rasterize_area_raw(raw, 2, 2)
        for py in range(2):
            for px in range(2):
                self.assertEqual(
                    res.area_at(px, py), Fraction(1, 2), (px, py)
                )
        self.assertEqual(res.total_area(), Fraction(2))


class ReferenceAgreementTests(unittest.TestCase):
    GLYPHS = [
        [_chain([(0, 0), (2, 0), (0, 1)])],
        [
            _chain([(0, 0), (4, 0), (4, 4), (0, 4)]),
            _chain([(1, 1), (1, 3), (3, 3), (3, 1)]),
        ],
        [
            _chain([(0, 0), (2, 0), (2, 2), (0, 2)]),
            _chain([(1, 1), (3, 1), (3, 3), (1, 3)]),
        ],
        [_chain([(-2, -2), (1, -2), (-2, 1)])],
        [_chain([(0, 0), (2, 2), (2, 0), (0, 2)])],
        [
            _chain(
                [
                    (0.5, 0.25),
                    (2.5, 0.75),
                    (2.25, 3.5),
                    (0.75, 2.5),
                ]
            )
        ],
    ]

    def test_pixel_areas_match_reference(self):
        for raw in self.GLYPHS:
            res = rasterize_area_raw(raw, 5, 5, origin=(-1, -1))
            for py in range(5):
                for px in range(5):
                    want = reference_pixel_area(raw, px - 1, py - 1)
                    self.assertEqual(
                        res.area_at(px, py), want, (raw, px, py)
                    )

    def test_pixel_sum_matches_clipped_fill_area(self):
        for raw in self.GLYPHS:
            res = rasterize_area_raw(raw, 5, 5, origin=(-1, -1))
            want = reference_window_area(raw, -1, -1, 5, 5)
            self.assertEqual(res.total_area(), want, raw)

    def test_random_polygons_match_reference(self):
        rng = random.Random(20260926)
        for _ in range(12):
            n = rng.randint(3, 7)
            pts = [
                (rng.randint(-16, 32) / 4.0,
                 rng.randint(-16, 32) / 4.0)
                for _ in range(n)
            ]
            raw = [_chain(pts)]
            res = rasterize_area_raw(raw, 6, 6, origin=(-2, -2))
            want_total = reference_window_area(raw, -2, -2, 6, 6)
            self.assertEqual(res.total_area(), want_total, raw)
            for py in range(6):
                for px in range(6):
                    want = reference_pixel_area(raw, px - 2, py - 2)
                    self.assertEqual(res.area_at(px, py), want, (raw, px, py))

    def test_origin_invariance(self):
        # The same pixel has the same area regardless of window origin.
        raw = self.GLYPHS[2]
        big = rasterize_area_raw(raw, 5, 5, origin=(-1, -1))
        small = rasterize_area_raw(raw, 3, 3, origin=(0, 0))
        for py in range(3):
            for px in range(3):
                self.assertEqual(
                    big.area_at(px + 1, py + 1), small.area_at(px, py)
                )


class ResultApiTests(unittest.TestCase):
    def setUp(self):
        self.raw = [_chain([(0, 0), (2, 0), (0, 1)])]
        self.res = rasterize_area_raw(self.raw, 2, 1)

    def test_result_type_and_fields(self):
        self.assertIsInstance(self.res, AreaCoverageResult)
        self.assertEqual((self.res.origin_x, self.res.origin_y), (0, 0))
        self.assertEqual((self.res.width, self.res.height), (2, 1))
        for row in self.res.areas:
            for a in row:
                self.assertIsInstance(a, Fraction)
                self.assertGreaterEqual(a, 0)
                self.assertLessEqual(a, 1)

    def test_coverage_at_is_float_view(self):
        self.assertEqual(self.res.coverage_at(0, 0), 0.75)
        self.assertEqual(self.res.coverage_at(1, 0), 0.25)

    def test_pixel_index_validation(self):
        for bad in ((-1, 0), (0, -1), (2, 0), (0, 1), (True, 0), (0.5, 0)):
            with self.assertRaises(InvalidInputError):
                self.res.area_at(*bad)


class RejectionTests(unittest.TestCase):
    def test_quad_segments_rejected(self):
        con = build_contour([(0, 0), (2, 0), (2, 2)], quads=((0, 1, -1),))
        with self.assertRaises(InvalidInputError) as cm:
            rasterize_area([con], 4, 4)
        self.assertIn("contours[0].segments[0]", str(cm.exception))

    def test_bad_origin(self):
        con = build_contour([(0, 0), (2, 0), (0, 2)])
        for bad in ((True, 0), (0.5, 0), (0,), "origin", (10**9, 0)):
            with self.assertRaises(InvalidInputError, msg=repr(bad)):
                rasterize_area([con], 2, 2, origin=bad)

    def test_bad_window(self):
        con = build_contour([(0, 0), (2, 0), (0, 2)])
        for w, h in ((0, 2), (2, 0), (65, 2), (True, 2)):
            with self.assertRaises(InvalidInputError, msg=(w, h)):
                rasterize_area([con], w, h)
        with self.assertRaises(InvalidInputError):
            rasterize_area([con], 2, 2, origin=(1024, 0))

    def test_bad_contours(self):
        con = build_contour([(0, 0), (2, 0), (0, 2)])
        for bad in ([], tuple(), [con] * 21, ["x"], None):
            with self.assertRaises(InvalidInputError, msg=repr(bad)):
                rasterize_area(bad, 2, 2)

    def test_invalid_raw_rejected_before_compute(self):
        with self.assertRaises(InvalidInputError):
            rasterize_area_raw(
                [_chain([(0, 0), (float("nan"), 1), (1, 1)])], 2, 2
            )


if __name__ == "__main__":
    unittest.main()
