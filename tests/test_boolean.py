"""Tests for the exact boolean set operations (union/intersection/difference).

Expected values never come from the code under test:

* areas are hand-computed exact rationals (shoelace / inclusion of
  simple polygons) or independently derived rectangle formulas;
* rasterized coverage is compared sub-sample by sub-sample against the
  pre-existing independent Fraction reference in
  :mod:`independent_reference`, which does not import the core;
* set-theoretic identities (inclusion-exclusion, De Morgan style
  differences) are checked exactly.
"""

from __future__ import annotations

import math
import random
import unittest
from fractions import Fraction

from winding_fill import (
    DIFFERENCE,
    INTERSECTION,
    UNION,
    InvalidInputError,
    Line,
    Contour,
    SelfIntersectionError,
    WindingFillError,
    boolean,
    boolean_raw,
    build_contour,
    difference,
    intersection,
    polygon_area,
    rasterize,
    union,
)

from independent_reference import reference_levels_lines


# ---------------------------------------------------------------------------
# Independent helpers (do NOT use the implementation to make expectations)
# ---------------------------------------------------------------------------


def rect(x0, y0, x1, y1):
    return build_contour([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def reversed_contour(c):
    return Contour(tuple(Line(s.p1, s.p0) for s in reversed(c.segments)))


def _f(x):
    return Fraction(x)


def poly_raw(poly):
    """Raw line segments of one exact rational polygon (open chain)."""
    return [
        {
            "type": "line",
            "p0": {"x": poly[i][0], "y": poly[i][1]},
            "p1": {
                "x": poly[(i + 1) % len(poly)][0],
                "y": poly[(i + 1) % len(poly)][1],
            },
        }
        for i in range(len(poly))
    ]


def contour_raw(c):
    return [
        {
            "type": "line",
            "p0": {"x": _f(s.p0.x), "y": _f(s.p0.y)},
            "p1": {"x": _f(s.p1.x), "y": _f(s.p1.y)},
        }
        for s in c.segments
    ]


def operand_raw(operand):
    raw = []
    for c in (operand if isinstance(operand, (list, tuple)) else [operand]):
        raw += contour_raw(c)
    return raw


def result_raw(result):
    raw = []
    for poly in result.polygons:
        raw += poly_raw(poly)
    return raw


def covered_subsamples(raw, w, h):
    """Independent exact per-subsample covered mask (Fraction reference)."""
    _, wind = reference_levels_lines(raw, w, h)
    return tuple(
        tuple(tuple(1 if v != 0 else 0 for v in wind[py][px])
              for px in range(w))
        for py in range(h)
    )


EMPTY_MASK_CACHE = {}


def empty_mask(w, h):
    key = (w, h)
    if key not in EMPTY_MASK_CACHE:
        EMPTY_MASK_CACHE[key] = tuple(
            tuple(tuple([0] * 16) for _ in range(w)) for _ in range(h)
        )
    return EMPTY_MASK_CACHE[key]


def combine(ma, mb, op, w, h):
    def g(a, b):
        if op == UNION:
            return tuple(x or y for x, y in zip(a, b))
        if op == INTERSECTION:
            return tuple(x and y for x, y in zip(a, b))
        return tuple(x and not y for x, y in zip(a, b))

    return tuple(
        tuple(g(ma[py][px], mb[py][px]) for px in range(w))
        for py in range(h)
    )


def assert_set_equals_raster(test, result, a, b, op, w, h):
    ma = covered_subsamples(operand_raw(a), w, h)
    mb = covered_subsamples(operand_raw(b), w, h)
    expected = combine(ma, mb, op, w, h)
    if result.polygons:
        got = covered_subsamples(result_raw(result), w, h)
    else:
        got = empty_mask(w, h)
    test.assertEqual(got, expected)


# ---------------------------------------------------------------------------
# Hand-computed areas
# ---------------------------------------------------------------------------


class HandComputedAreaTests(unittest.TestCase):
    def test_polygon_area_sign_and_exactness(self):
        ccw = rect(0, 0, 3, 4)
        cw = reversed_contour(ccw)
        self.assertEqual(polygon_area(ccw), Fraction(12))
        self.assertEqual(polygon_area(cw), Fraction(-12))
        with self.assertRaises(InvalidInputError):
            polygon_area("not a contour")
        curved = build_contour([(0, 0), (4, 0), (4, 4), (0, 4)],
                               quads=((0, 2, -1),))
        with self.assertRaises(InvalidInputError):
            polygon_area(curved)

    def test_overlapping_squares(self):
        a = rect(1, 1, 5, 5)          # area 16
        b = rect(3, 3, 7, 7)          # area 16, overlap 2x2 = 4
        self.assertEqual(union(a, b).area, Fraction(28))
        self.assertEqual(intersection(a, b).area, Fraction(4))
        self.assertEqual(difference(a, b).area, Fraction(12))
        # union is one simple 8-vertex corner octagon
        u = union(a, b)
        self.assertEqual(len(u.contours), 1)
        self.assertEqual(len(u.contours[0].segments), 8)
        for r in (union(a, b), intersection(a, b), difference(a, b)):
            self.assertEqual(r.area, sum(r.contour_areas, Fraction(0)))

    def test_containment_produces_oriented_hole(self):
        outer = rect(1, 1, 9, 9)      # 64
        inner = rect(3, 3, 7, 7)      # 16, fully inside
        self.assertEqual(union(outer, inner).area, Fraction(64))
        self.assertEqual(len(union(outer, inner).contours), 1)
        self.assertEqual(intersection(outer, inner).area, Fraction(16))
        d = difference(outer, inner)
        self.assertEqual(d.area, Fraction(48))
        self.assertEqual(len(d.contours), 2)
        # outer boundary CCW (positive), hole CW (negative)
        self.assertEqual(d.contour_areas, (Fraction(64), Fraction(-16)))
        outer_loop, hole_loop = d.contours
        self.assertGreater(polygon_area(outer_loop), 0)
        self.assertLess(polygon_area(hole_loop), 0)

    def test_disjoint_and_empty(self):
        a = rect(0, 0, 2, 2)
        b = rect(5, 5, 7, 7)
        u = union(a, b)
        self.assertEqual(u.area, Fraction(8))
        self.assertEqual(len(u.contours), 2)
        i = intersection(a, b)
        self.assertTrue(i.is_empty)
        self.assertEqual(i.contours, ())
        self.assertEqual(i.area, 0)
        self.assertEqual(difference(a, b).area, Fraction(4))

    def test_identical_operands(self):
        a = rect(1, 1, 5, 5)
        self.assertEqual(union(a, a).area, Fraction(16))
        self.assertEqual(intersection(a, a).area, Fraction(16))
        self.assertTrue(difference(a, a).is_empty)
        self.assertEqual(len(union(a, a).contours), 1)
        self.assertEqual(len(intersection(a, a).contours), 1)

    def test_triangles_with_thirds_intersection(self):
        # A: right triangle under y = 6 - 2x, vertices (0,0),(3,0),(0,6),
        #    area 9.
        # B: right triangle under y = 4 - x/3, vertices (0,0),(12,0),
        #    (0,4), area 24.
        # The hypotenuses meet at 6-2x = 4-x/3  =>  x = 6/5, y = 18/5.
        # Inside A for x in [0,3]: the overlap top is the lower line.
        # For x in [0,6/5] B is lower (4-x/3), for x in [6/5,3] A is
        # lower (6-2x); integrating gives 102/25 + 93/25 = 39/5.
        # Union = 9 + 24 - 39/5 = 126/5; A\\B = 9 - 39/5 = 6/5.
        a = build_contour([(0, 0), (3, 0), (0, 6)])
        b = build_contour([(0, 0), (12, 0), (0, 4)])
        self.assertEqual(intersection(a, b).area, Fraction(39, 5))
        self.assertEqual(union(a, b).area, Fraction(126, 5))
        self.assertEqual(difference(a, b).area, Fraction(6, 5))
        # the genuine rational vertex is present, unrounded, in polygons
        verts = {p for poly in intersection(a, b).polygons for p in poly}
        self.assertIn((Fraction(6, 5), Fraction(18, 5)), verts)


# ---------------------------------------------------------------------------
# Tangency, coincident edges and containment
# ---------------------------------------------------------------------------


class DegenerateContactTests(unittest.TestCase):
    def test_vertex_tangency_two_squares(self):
        a = rect(0, 0, 2, 2)
        b = rect(2, 2, 4, 4)        # touch at exactly one point (2,2)
        u = union(a, b)
        self.assertEqual(u.area, Fraction(8))
        self.assertEqual(len(u.contours), 2)   # loops stay separate
        self.assertTrue(intersection(a, b).is_empty)
        self.assertEqual(difference(a, b).area, Fraction(4))
        assert_set_equals_raster(self, u, a, b, UNION, 6, 6)

    def test_triangle_apex_touches_edge_interior(self):
        a = rect(0, 0, 6, 5)
        b = build_contour([(3, 5), (5, 9), (1, 9)])  # apex on a's top edge
        u = union(a, b)
        self.assertEqual(u.area, Fraction(38))       # 30 + triangle 8
        self.assertEqual(len(u.contours), 2)
        self.assertTrue(intersection(a, b).is_empty)
        assert_set_equals_raster(self, u, a, b, UNION, 10, 10)

    def test_full_shared_edge_merges_into_rectangle(self):
        a = rect(0, 0, 2, 2)
        b = rect(2, 0, 4, 2)
        u = union(a, b)
        self.assertEqual(u.area, Fraction(8))
        self.assertEqual(len(u.contours), 1)
        self.assertEqual(len(u.contours[0].segments), 4)
        self.assertTrue(intersection(a, b).is_empty)  # zero-area contact
        self.assertEqual(difference(a, b).area, Fraction(4))

    def test_partial_coincident_edge(self):
        # B sits on A's top edge, sharing only the segment x in [2,4].
        a = rect(0, 0, 4, 4)
        b = rect(2, 4, 6, 8)
        u = union(a, b)
        self.assertEqual(u.area, Fraction(32))
        self.assertTrue(intersection(a, b).is_empty)
        assert_set_equals_raster(self, u, a, b, UNION, 9, 9)

    def test_t_contact_vertex_on_edge(self):
        # B's whole bottom edge lies on A's top edge (both B vertices are
        # interior points of A's edge).
        a = rect(0, 0, 8, 4)
        b = rect(2, 4, 6, 8)
        u = union(a, b)
        self.assertEqual(u.area, Fraction(48))
        self.assertEqual(len(u.contours), 1)
        assert_set_equals_raster(self, u, a, b, UNION, 9, 9)

    def test_hole_touching_outer_boundary_at_one_vertex(self):
        outer = rect(0, 0, 8, 8)
        notch = build_contour([(4, 0), (5, 3), (3, 3)])  # apex on edge y=0
        d = difference(outer, notch)
        self.assertEqual(d.area, Fraction(61))   # 64 - triangle 3
        self.assertEqual(d.contour_areas, (Fraction(64), Fraction(-3)))
        assert_set_equals_raster(self, d, outer, notch, DIFFERENCE, 9, 9)

    def test_interior_contour_touching_from_inside(self):
        outer = rect(0, 0, 8, 8)
        inner = build_contour([(4, 0), (6, 2), (2, 2)])  # touches at (4,0)
        self.assertEqual(union(outer, inner).area, Fraction(64))
        self.assertEqual(intersection(outer, inner).area, Fraction(4))
        assert_set_equals_raster(
            self, intersection(outer, inner), outer, inner,
            INTERSECTION, 9, 9
        )


# ---------------------------------------------------------------------------
# Exact rasterized-set equivalence against the independent reference
# ---------------------------------------------------------------------------


class RasterizationEquivalenceTests(unittest.TestCase):
    def test_concave_arrow_vs_square_all_ops(self):
        c = build_contour(
            [(0, 1), (4, 1), (6, 0), (8, 1), (8, 7), (4, 5), (0, 7)]
        )
        b = rect(3, 2, 10, 6)
        for op, fn in ((UNION, union), (INTERSECTION, intersection),
                       (DIFFERENCE, difference)):
            with self.subTest(op=op):
                assert_set_equals_raster(self, fn(c, b), c, b, op, 11, 9)

    def test_random_rectangles_exact(self):
        rng = random.Random(20260924)
        for _ in range(60):
            x0 = rng.choice([0, 1, 2, 3, 5, 6, 7])
            y0 = rng.choice([0, 1, 2, 3, 5, 6, 7])
            a = rect(x0, y0, x0 + rng.choice([2, 3, 4]),
                     y0 + rng.choice([2, 3, 4]))
            b = rect(x0 + rng.choice([0, 1, 2]), y0 + rng.choice([0, 1, 2]),
                     x0 + rng.choice([5, 6, 7]), y0 + rng.choice([5, 6, 7]))
            for op, fn in ((UNION, union), (INTERSECTION, intersection),
                           (DIFFERENCE, difference)):
                with self.subTest(op=op):
                    assert_set_equals_raster(self, fn(a, b), a, b, op, 12, 12)

    def test_random_triangles_exact(self):
        rng = random.Random(4242)
        w = h = 11
        for _ in range(40):
            while True:
                pts = [(rng.choice([0, 2, 4, 6, 8]),
                        rng.choice([0, 2, 4, 6, 8])) for _ in range(3)]
                (x0, y0), (x1, y1), (x2, y2) = pts
                if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
                    break
            a = build_contour(pts)
            b = rect(rng.choice([0, 2, 4, 6]), rng.choice([0, 2, 4, 6]),
                     9, 9)
            for op, fn in ((UNION, union), (INTERSECTION, intersection),
                           (DIFFERENCE, difference)):
                with self.subTest(op=op, pts=pts):
                    assert_set_equals_raster(self, fn(a, b), a, b, op, w, h)

    def test_inclusion_exclusion_of_areas(self):
        # |A u B| + |A n B| = |A| + |B|, areas derived independently by
        # the rectangle overlap formula (not by the implementation).
        rng = random.Random(777)
        for _ in range(50):
            ax0, ay0 = rng.randint(0, 6), rng.randint(0, 6)
            ax1, ay1 = ax0 + rng.randint(2, 5), ay0 + rng.randint(2, 5)
            bx0, by0 = rng.randint(0, 6), rng.randint(0, 6)
            bx1, by1 = bx0 + rng.randint(2, 5), by0 + rng.randint(2, 5)
            a, b = rect(ax0, ay0, ax1, ay1), rect(bx0, by0, bx1, by1)
            area_a = Fraction((ax1 - ax0) * (ay1 - ay0))
            area_b = Fraction((bx1 - bx0) * (by1 - by0))
            ov_x = max(0, min(ax1, bx1) - max(ax0, bx0))
            ov_y = max(0, min(ay1, by1) - max(ay0, by0))
            ov = Fraction(ov_x * ov_y)
            self.assertEqual(union(a, b).area + intersection(a, b).area,
                             area_a + area_b)
            self.assertEqual(intersection(a, b).area, ov)
            self.assertEqual(difference(a, b).area, area_a - ov)

    def test_de_morgan_style_partition(self):
        # For every sub-sample: A = (A\\B) disjoint-union (A n B).
        rng = random.Random(31337)
        w = h = 11
        for _ in range(25):
            a = rect(rng.randint(0, 5), rng.randint(0, 5),
                     rng.randint(7, 10), rng.randint(7, 10))
            b = rect(rng.randint(0, 7), rng.randint(0, 7),
                     rng.randint(8, 11), rng.randint(8, 11))
            md = covered_subsamples(result_raw(difference(a, b)), w, h)
            mi = covered_subsamples(result_raw(intersection(a, b)), w, h)
            ma = covered_subsamples(operand_raw(a), w, h)
            for py in range(h):
                for px in range(w):
                    for s in range(16):
                        # disjoint
                        self.assertFalse(md[py][px][s] and mi[py][px][s])
                        # cover A exactly
                        self.assertEqual(
                            md[py][px][s] or mi[py][px][s],
                            bool(ma[py][px][s]),
                        )

    def test_multi_contour_operands(self):
        a = [rect(0, 0, 2, 4), rect(6, 0, 8, 4)]
        b = rect(1, 1, 7, 3)
        self.assertEqual(union(a, b).area, Fraction(24))
        self.assertEqual(intersection(a, b).area, Fraction(4))
        self.assertEqual(difference(a, b).area, Fraction(12))
        assert_set_equals_raster(self, union(a, b), a, b, UNION, 9, 5)
        assert_set_equals_raster(self, difference(a, b), a, b,
                                 DIFFERENCE, 9, 5)


# ---------------------------------------------------------------------------
# Orientation invariance and round-tripping
# ---------------------------------------------------------------------------


class OrientationAndChainingTests(unittest.TestCase):
    def test_result_invariant_under_operand_reversal(self):
        a, b = rect(1, 1, 6, 6), rect(4, 4, 9, 9)
        for fn in (union, intersection, difference):
            areas = {
                fn(x, y).area
                for x in (a, reversed_contour(a))
                for y in (b, reversed_contour(b))
            }
            self.assertEqual(len(areas), 1, fn.__name__)

    def test_result_loops_feed_back_into_pipeline(self):
        ring = difference(rect(0, 0, 6, 6), rect(2, 2, 4, 4))  # area 32
        again = difference(list(ring.contours), [rect(1, 1, 3, 3)])
        # 2x2 square [1,3]^2 removes area 4 except its overlap with the
        # existing [2,4]^2 hole, which is the unit square [2,3]^2.
        self.assertEqual(again.area, Fraction(29))

    def test_result_rasterizes_and_empty_gives_zero_canvas(self):
        r = difference(rect(0, 0, 4, 4), rect(1, 1, 3, 3))
        result = r.rasterize(6, 6)
        direct = rasterize(list(r.contours), 6, 6)
        self.assertEqual(result.levels, direct.levels)
        # the hole is empty, outer region solid
        self.assertEqual(result.levels[2][2], 0)
        self.assertEqual(result.levels[0][0], 16)
        empty = intersection(rect(0, 0, 1, 1), rect(3, 3, 4, 4)).rasterize(4, 4)
        self.assertTrue(all(v == 0 for row in empty.levels for v in row))

    def test_boolean_raw_matches_typed_api(self):
        raw_a = [[
            {"type": "line", "p0": {"x": 0, "y": 0}, "p1": {"x": 4, "y": 0}},
            {"type": "line", "p0": {"x": 4, "y": 0}, "p1": {"x": 4, "y": 4}},
            {"type": "line", "p0": {"x": 4, "y": 4}, "p1": {"x": 0, "y": 4}},
            {"type": "line", "p0": {"x": 0, "y": 4}, "p1": {"x": 0, "y": 0}},
        ]]
        raw_b = [[
            {"type": "line", "p0": {"x": 2, "y": 2}, "p1": {"x": 6, "y": 2}},
            {"type": "line", "p0": {"x": 6, "y": 2}, "p1": {"x": 6, "y": 6}},
            {"type": "line", "p0": {"x": 6, "y": 6}, "p1": {"x": 2, "y": 6}},
            {"type": "line", "p0": {"x": 2, "y": 6}, "p1": {"x": 2, "y": 2}},
        ]]
        typed = difference(build_contour([(0, 0), (4, 0), (4, 4), (0, 4)]),
                           build_contour([(2, 2), (6, 2), (6, 6), (2, 6)]))
        raw = boolean_raw(raw_a, raw_b, DIFFERENCE)
        self.assertEqual(raw.area, typed.area)
        self.assertEqual(raw.polygons, typed.polygons)


# ---------------------------------------------------------------------------
# Rejection of invalid / non-simple input
# ---------------------------------------------------------------------------


class RejectionTests(unittest.TestCase):
    def test_self_intersecting_bowtie_rejected(self):
        bowtie = build_contour([(0, 0), (4, 4), (4, 0), (0, 4)])
        with self.assertRaises(SelfIntersectionError) as cm:
            union(bowtie, rect(6, 6, 8, 8))
        self.assertIn("a[0]", cm.exception.location)
        # hierarchy: a WindingFillError and a ValueError
        self.assertIsInstance(cm.exception, WindingFillError)
        self.assertIsInstance(cm.exception, ValueError)

    def test_self_intersection_of_second_operand_located_on_b(self):
        bowtie = build_contour([(0, 0), (4, 4), (4, 0), (0, 4)])
        with self.assertRaises(SelfIntersectionError) as cm:
            difference(rect(0, 0, 8, 8), bowtie)
        self.assertIn("b[0]", cm.exception.location)

    def test_repeated_vertex_rejected(self):
        # vertex (2,2) occurs twice non-adjacently
        pinched = build_contour(
            [(0, 0), (4, 0), (4, 4), (2, 2), (0, 4), (2, 2)]
        )
        with self.assertRaises(SelfIntersectionError):
            intersection(pinched, rect(0, 0, 5, 5))

    def test_adjacent_edge_overlap_rejected(self):
        # (0,0)->(4,0)->(2,0) doubles back along the same line
        spike = build_contour([(0, 0), (4, 0), (2, 0), (2, 4)])
        with self.assertRaises(SelfIntersectionError):
            union(spike, rect(6, 6, 8, 8))

    def test_quadratic_contour_rejected(self):
        curved = build_contour([(0, 0), (4, 0), (4, 4), (0, 4)],
                               quads=((0, 2, -1),))
        with self.assertRaises(InvalidInputError):
            union(curved, rect(6, 6, 8, 8))

    def test_bad_operation_name(self):
        for bad in ("xor", "UNION", 7, None):
            with self.assertRaises(InvalidInputError):
                boolean(rect(0, 0, 2, 2), rect(1, 1, 3, 3), bad)

    def test_bad_operand_types_and_counts(self):
        good = rect(0, 0, 2, 2)
        for bad in ("x", 5, None, [], {}):
            with self.assertRaises((InvalidInputError, WindingFillError)):
                union(bad, good)
        with self.assertRaises(InvalidInputError):
            union([good, "not-a-contour"], good)
        many = [rect(k, k, k + 1, k + 1) for k in range(21)]
        with self.assertRaises(InvalidInputError):
            union(many, good)

    def test_raw_nan_rejected_before_computation(self):
        raw = [[
            {"type": "line", "p0": {"x": 0, "y": 0},
             "p1": {"x": math.nan, "y": 0}},
            {"type": "line", "p0": {"x": math.nan, "y": 0},
             "p1": {"x": 4, "y": 4}},
            {"type": "line", "p0": {"x": 4, "y": 4},
             "p1": {"x": 0, "y": 0}},
        ]]
        good = [[
            {"type": "line", "p0": {"x": 1, "y": 1}, "p1": {"x": 2, "y": 1}},
            {"type": "line", "p0": {"x": 2, "y": 1}, "p1": {"x": 2, "y": 2}},
            {"type": "line", "p0": {"x": 2, "y": 2}, "p1": {"x": 1, "y": 2}},
            {"type": "line", "p0": {"x": 1, "y": 2}, "p1": {"x": 1, "y": 1}},
        ]]
        with self.assertRaises(InvalidInputError) as cm:
            boolean_raw(raw, good, UNION)
        self.assertTrue(cm.exception.location.startswith("a."))
        # the second operand's faults are located on side b
        with self.assertRaises(InvalidInputError) as cm2:
            boolean_raw(good, raw, UNION)
        self.assertTrue(cm2.exception.location.startswith("b."))

    def test_raw_self_intersection_rejected(self):
        bowtie_raw = [[
            {"type": "line", "p0": {"x": 0, "y": 0},
             "p1": {"x": 4, "y": 4}},
            {"type": "line", "p0": {"x": 4, "y": 4},
             "p1": {"x": 4, "y": 0}},
            {"type": "line", "p0": {"x": 4, "y": 0},
             "p1": {"x": 0, "y": 4}},
            {"type": "line", "p0": {"x": 0, "y": 4},
             "p1": {"x": 0, "y": 0}},
        ]]
        good = [[
            {"type": "line", "p0": {"x": 6, "y": 6}, "p1": {"x": 8, "y": 6}},
            {"type": "line", "p0": {"x": 8, "y": 6}, "p1": {"x": 8, "y": 8}},
            {"type": "line", "p0": {"x": 8, "y": 8}, "p1": {"x": 6, "y": 8}},
            {"type": "line", "p0": {"x": 6, "y": 8}, "p1": {"x": 6, "y": 6}},
        ]]
        with self.assertRaises(SelfIntersectionError) as cm:
            boolean_raw(bowtie_raw, good, UNION)
        self.assertIn("a[0]", cm.exception.location)

    def test_failure_leaves_operands_unmodified(self):
        bowtie = build_contour([(0, 0), (4, 4), (4, 0), (0, 4)])
        snapshot = tuple(
            (s.p0.x, s.p0.y, s.p1.x, s.p1.y) for s in bowtie.segments
        )
        with self.assertRaises(SelfIntersectionError):
            union(bowtie, rect(8, 8, 9, 9))
        self.assertEqual(
            tuple((s.p0.x, s.p0.y, s.p1.x, s.p1.y)
                  for s in bowtie.segments),
            snapshot,
        )

    def test_result_loops_are_simple_closed_contours(self):
        rng = random.Random(9001)
        for _ in range(60):
            a = rect(rng.randint(0, 5), rng.randint(0, 5),
                     rng.randint(7, 10), rng.randint(7, 10))
            b = rect(rng.randint(0, 7), rng.randint(0, 7),
                     rng.randint(8, 11), rng.randint(8, 11))
            for fn in (union, intersection, difference):
                r = fn(a, b)
                for c in r.contours:
                    self.assertGreaterEqual(len(c.segments), 3)
                    self.assertEqual(c.segments[-1].p1, c.segments[0].p0)
                    for s in c.segments:
                        self.assertNotEqual(s.p0, s.p1)


if __name__ == "__main__":
    unittest.main()
