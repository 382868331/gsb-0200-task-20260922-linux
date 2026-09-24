"""Tests for the two-polygon boolean operations (union / intersection /
difference).

Expected values never come from the implementation under test:

* areas are checked by hand (simple Fractions) and by the analytic
  inclusion-exclusion identity
  ``area(union) = area(A) + area(B) - area(intersection)`` and
  ``area(difference) = area(A) - area(intersection)``;
* rasterized coverage of every result is checked, per sub-sample,
  against membership decisions made by the independent exact-Fraction
  winding oracle in ``independent_reference.py`` (which does not import
  the boolean module or the rasterizer core);
* random inputs are convex hulls of deterministic random point sets,
  so they are guaranteed simple without consulting the implementation.
"""

from __future__ import annotations

import random
import unittest
from fractions import Fraction

from winding_fill import (
    DIFFERENCE,
    INTERSECTION,
    UNION,
    Contour,
    InvalidInputError,
    Line,
    Quad,
    Point,
    boolean_difference,
    boolean_intersection,
    boolean_op_raw,
    boolean_union,
    build_contour,
    rasterize,
    signed_area,
)

from independent_reference import winding_point_lines


# ---------------------------------------------------------------------------
# Independent helpers (no use of the boolean implementation)
# ---------------------------------------------------------------------------


def contour_to_raw(contour):
    raw = []
    for seg in contour.segments:
        d0 = {"x": seg.p0.x, "y": seg.p0.y}
        d1 = {"x": seg.p1.x, "y": seg.p1.y}
        if isinstance(seg, Line):
            raw.append({"type": "line", "p0": d0, "p1": d1})
        else:
            raw.append(
                {"type": "quad", "p0": d0,
                 "c": {"x": seg.c.x, "y": seg.c.y}, "p1": d1}
            )
    return raw


def reverse_contour(contour):
    rev = []
    for seg in reversed(contour.segments):
        if isinstance(seg, Line):
            rev.append(Line(seg.p1, seg.p0))
        else:
            rev.append(Quad(seg.p1, seg.c, seg.p0))
    return Contour(tuple(rev))


def convex_hull(points):
    """Andrew's monotone chain (independent code; Fraction tuples)."""
    pts = sorted(set(points))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def hull_contour(rng, nlo, nhi, scale=8, den=8):
    n = rng.randint(nlo, nhi)
    pts = [(rng.randint(1, scale * den), rng.randint(1, scale * den))
           for _ in range(n * 3)]
    pts = [(Fraction(x, den), Fraction(y, den)) for x, y in pts]
    hull = convex_hull(pts)
    return build_contour([(float(x), float(y)) for x, y in hull])


def expected_membership(raw_a, raw_b, op, w, h):
    """Independent per-sub-sample membership grid via Fraction ray casts."""
    grid = [[[False] * 16 for _ in range(w)] for _ in range(h)]
    for py in range(h):
        for j in range(4):
            sy = Fraction(py) + Fraction(2 * j + 1, 8)
            for px in range(w):
                for i in range(4):
                    sx = Fraction(px) + Fraction(2 * i + 1, 8)
                    wa = winding_point_lines([raw_a], sx, sy)
                    wb = winding_point_lines([raw_b], sx, sy)
                    ina, inb = wa != 0, wb != 0
                    if op == UNION:
                        inside = ina or inb
                    elif op == INTERSECTION:
                        inside = ina and inb
                    else:
                        inside = ina and not inb
                    grid[py][px][j * 4 + i] = inside
    return grid


def _point_seg_dist2(q, a, b):
    """Exact squared distance from Fraction point q to segment [a, b]."""
    ab = (b[0] - a[0], b[1] - a[1])
    aq = (q[0] - a[0], q[1] - a[1])
    den = ab[0] * ab[0] + ab[1] * ab[1]
    t = (aq[0] * ab[0] + aq[1] * ab[1]) / den
    t = max(Fraction(0), min(Fraction(1), t))
    px, py = a[0] + t * ab[0], a[1] + t * ab[1]
    return (q[0] - px) ** 2 + (q[1] - py) ** 2


def _raw_edges_fraction(raw):
    out = []
    for seg in raw:
        out.append(
            (
                (Fraction(seg["p0"]["x"]), Fraction(seg["p0"]["y"])),
                (Fraction(seg["p1"]["x"]), Fraction(seg["p1"]["y"])),
            )
        )
    return out


# Float output boundary agreement only has to hold for samples far from
# the exact boundary; exact rational intersection points (e.g. 23/3)
# round to double at the same ~1e-15 relative scale documented for the
# rasterizer core.
_BOUNDARY_EPS = Fraction(1, 10**9)


def assert_matches_grid(test, result_contours, raw_a, raw_b, op, w, h):
    expected = expected_membership(raw_a, raw_b, op, w, h)
    if result_contours:
        raster = rasterize(list(result_contours), w, h)
    edges = _raw_edges_fraction(raw_a) + _raw_edges_fraction(raw_b)
    near_misses = 0
    for py in range(h):
        for px in range(w):
            exp_winds = tuple(1 if v else 0 for v in expected[py][px])
            if result_contours:
                got_winds = tuple(1 if x != 0 else 0
                                  for x in raster.windings[py][px])
            else:
                got_winds = tuple(0 for _ in expected[py][px])
                test.assertFalse(
                    any(expected[py][px]),
                    f"{op}: exact result is empty but pixel ({px},{py}) "
                    "is covered by the oracle",
                )
            if got_winds == exp_winds:
                if result_contours:
                    test.assertEqual(
                        raster.levels[py][px], sum(expected[py][px])
                    )
                continue
            pixel_misses = 0
            # Every disagreement must be a sub-sample so close to the
            # exact input boundary that 1-ulp float rounding can flip
            # it; geometrically (area-wise) the result is still exact.
            for j in range(4):
                for i in range(4):
                    idx = j * 4 + i
                    if got_winds[idx] == exp_winds[idx]:
                        continue
                    q = (
                        Fraction(px) + Fraction(2 * i + 1, 8),
                        Fraction(py) + Fraction(2 * j + 1, 8),
                    )
                    d2 = min(_point_seg_dist2(q, a, b) for a, b in edges)
                    test.assertLess(
                        d2, _BOUNDARY_EPS**2,
                        f"{op}: mismatch at pixel ({px},{py}) sample "
                        f"[{i},{j}] with distance {float(d2**0.5):.3e} "
                        "from the exact boundary (float epsilon exceeded)",
                    )
                    pixel_misses += 1
                    near_misses += 1
            if result_contours:
                # Any level difference is fully explained by boundary flips.
                test.assertLessEqual(
                    abs(raster.levels[py][px] - sum(expected[py][px])),
                    pixel_misses,
                )
    return near_misses


# ---------------------------------------------------------------------------
# Hand-computed cases
# ---------------------------------------------------------------------------


def square(x0, y0, x1, y1):
    return build_contour([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


class HandComputedBooleanTests(unittest.TestCase):
    def setUp(self):
        # Two axis-aligned squares overlapping in a 2x2 block.
        self.a = square(0, 0, 4, 4)          # area 16
        self.b = square(2, 2, 6, 6)          # area 16

    def test_areas_hand_computed(self):
        self.assertEqual(boolean_union(self.a, self.b).area, Fraction(28))
        self.assertEqual(
            boolean_intersection(self.a, self.b).area, Fraction(4)
        )
        self.assertEqual(
            boolean_difference(self.a, self.b).area, Fraction(12)
        )

    def test_inclusion_exclusion_identity(self):
        ra = signed_area(self.a)
        rb = signed_area(self.b)
        u = boolean_union(self.a, self.b)
        i = boolean_intersection(self.a, self.b)
        d = boolean_difference(self.a, self.b)
        self.assertEqual(u.area, ra + rb - i.area)
        self.assertEqual(d.area, ra - i.area)
        self.assertEqual(boolean_difference(self.b, self.a).area, rb - i.area)

    def test_union_polygon_vertices_hand_checked(self):
        u = boolean_union(self.a, self.b)
        self.assertEqual(len(u.contours), 1)
        pts = [(s.p0.x, s.p0.y) for s in u.contours[0].segments]
        self.assertEqual(
            pts,
            [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (6.0, 2.0),
             (6.0, 6.0), (2.0, 6.0), (2.0, 4.0), (0.0, 4.0)],
        )

    def test_result_loops_oriented_ccw_outer_cw_hole(self):
        d = boolean_difference(square(0, 0, 8, 8), square(2, 2, 6, 6))
        self.assertEqual(len(d.contours), 2)
        areas = [signed_area(c) for c in d.contours]
        self.assertEqual(areas, [Fraction(64), Fraction(-16)])
        self.assertEqual(d.area, Fraction(48))

    def test_result_is_plain_contours_accepted_by_rasterize(self):
        u = boolean_union(self.a, self.b)
        result = rasterize(list(u.contours), 8, 8)
        # pixel (1,1) is inside A only -> solid; (5,5) inside B only
        self.assertEqual(result.levels[1][1], 16)
        self.assertEqual(result.levels[5][5], 16)
        # pixel (3,3) in the overlap -> solid; outside -> empty
        self.assertEqual(result.levels[3][3], 16)
        self.assertEqual(result.levels[7][7], 0)

    def test_orientation_of_inputs_does_not_matter(self):
        ar = reverse_contour(self.a)
        br = reverse_contour(self.b)
        self.assertEqual(signed_area(ar), -Fraction(16))
        for op, fn in (
            (UNION, boolean_union),
            (INTERSECTION, boolean_intersection),
            (DIFFERENCE, boolean_difference),
        ):
            r1 = fn(self.a, self.b)
            r2 = fn(ar, br) if op != DIFFERENCE else fn(ar, self.b)
            self.assertEqual(r1.area, r2.area, op)
            self.assertEqual(
                sorted([(s.p0.x, s.p0.y) for s in c.segments]
                       for c in r1.contours),
                sorted([(s.p0.x, s.p0.y) for s in c.segments]
                       for c in r2.contours),
                op,
            )

    def test_rational_coordinates_exact(self):
        # Two right triangles with thirds; all areas are clean fractions.
        t1 = build_contour([(0, 0), (3, 0), (0, 3)])
        t2 = build_contour([(1, 0), (4, 0), (1, 3)])
        self.assertEqual(boolean_intersection(t1, t2).area, Fraction(2))
        self.assertEqual(boolean_union(t1, t2).area, Fraction(7))
        self.assertEqual(boolean_difference(t1, t2).area, Fraction(5, 2))
        # float 1/3 inputs are exact binary rationals end to end
        tri = build_contour([(0.0, 0.0), (1.0, 0.0), (0.0, 3.0)])
        self.assertEqual(signed_area(tri), Fraction(3, 2))


# ---------------------------------------------------------------------------
# Degenerate relations: tangency, coincident edges, containment, disjoint
# ---------------------------------------------------------------------------


class DegenerateRelationTests(unittest.TestCase):
    def setUp(self):
        self.a = square(0, 0, 2, 2)

    def test_disjoint_polygons(self):
        b = square(3, 0, 5, 2)
        u = boolean_union(self.a, b)
        self.assertEqual(len(u.contours), 2)
        self.assertEqual(u.area, Fraction(8))
        self.assertEqual(boolean_intersection(self.a, b).contours, ())
        self.assertEqual(boolean_intersection(self.a, b).area, Fraction(0))
        d = boolean_difference(self.a, b)
        self.assertEqual(d.area, Fraction(4))

    def test_edge_to_edge_tangency(self):
        b = square(2, 0, 4, 2)  # shares the whole right edge of A
        u = boolean_union(self.a, b)
        self.assertEqual(u.area, Fraction(8))
        self.assertEqual(len(u.contours), 1)
        # The two rectangles fuse into a plain rectangle: the T vertices
        # on the shared line are collinear and are merged away.
        self.assertEqual(len(u.contours[0].segments), 4)
        i = boolean_intersection(self.a, b)
        self.assertEqual(i.area, Fraction(0))
        self.assertEqual(i.contours, ())
        self.assertEqual(boolean_difference(self.a, b).area, Fraction(4))

    def test_partial_coincident_edge(self):
        # B straddles A's bottom: overlap of boundary along [1,2] x {0}.
        b = square(1, -2, 3, 2)
        self.assertEqual(boolean_union(self.a, b).area, Fraction(10))
        self.assertEqual(boolean_intersection(self.a, b).area, Fraction(2))
        self.assertEqual(boolean_difference(self.a, b).area, Fraction(2))

    def test_corner_to_corner_tangency_two_components(self):
        b = square(2, 2, 4, 4)  # touches A only at (2,2)
        u = boolean_union(self.a, b)
        self.assertEqual(u.area, Fraction(8))
        self.assertEqual(len(u.contours), 2)  # point contact splits
        self.assertEqual(boolean_intersection(self.a, b).area, Fraction(0))

    def test_vertex_to_edge_tangency(self):
        a = build_contour([(0, 0), (2, 0), (2, 4), (0, 4)])
        tri = build_contour([(2, 2), (4, 1), (4, 3)])
        u = boolean_union(a, tri)
        self.assertEqual(u.area, signed_area(a) + signed_area(tri))
        self.assertEqual(len(u.contours), 2)  # touch at one point only
        self.assertEqual(boolean_intersection(a, tri).area, Fraction(0))
        self.assertEqual(
            boolean_difference(a, tri).area, signed_area(a)
        )

    def test_containment(self):
        outer = square(0, 0, 8, 8)
        inner = square(2, 2, 6, 6)
        self.assertEqual(boolean_union(outer, inner).area, Fraction(64))
        self.assertEqual(
            boolean_intersection(outer, inner).area, Fraction(16)
        )
        d = boolean_difference(outer, inner)
        self.assertEqual(d.area, Fraction(48))
        self.assertEqual(len(d.contours), 2)  # outer + hole
        # inner minus outer is the empty region
        self.assertEqual(
            boolean_difference(inner, outer).contours, ()
        )
        # the ring really rasterizes with an empty center
        raster = rasterize(list(d.contours), 10, 10)
        self.assertEqual(raster.levels[1][1], 16)
        self.assertEqual(raster.levels[4][4], 0)
        self.assertTrue(all(w == 1 for w in raster.windings[1][1]))
        self.assertTrue(all(w == 0 for w in raster.windings[4][4]))

    def test_identical_polygons(self):
        self.assertEqual(boolean_union(self.a, self.a).area, Fraction(4))
        self.assertEqual(
            boolean_intersection(self.a, self.a).area, Fraction(4)
        )
        self.assertEqual(boolean_difference(self.a, self.a).area, Fraction(0))
        self.assertEqual(boolean_difference(self.a, self.a).contours, ())

    def test_same_polygon_extra_collinear_vertex(self):
        # Same square, one side split by a collinear vertex.
        a2 = build_contour(
            [(0, 0), (1, 0), (2, 0), (2, 2), (0, 2)]
        )
        self.assertEqual(boolean_union(self.a, a2).area, Fraction(4))
        self.assertEqual(
            boolean_intersection(self.a, a2).area, Fraction(4)
        )
        self.assertEqual(
            boolean_difference(self.a, a2).contours, ()
        )

    def test_tangent_t_junction_rasterized(self):
        # Square B touches A's right edge with a corner at its midpoint;
        # verify the whole picture against the independent oracle.
        a = square(0, 0, 4, 4)
        b = square(4, 1, 6, 3)
        raw_a = contour_to_raw(a)
        raw_b = contour_to_raw(b)
        u = boolean_union(a, b)
        assert_matches_grid(self, u.contours, raw_a, raw_b, UNION, 8, 5)
        i = boolean_intersection(a, b)
        self.assertEqual(i.area, 0)

    def test_inner_polygon_sharing_outer_edge(self):
        # B lies inside A except that its top edge lies on A's top edge.
        a = square(0, 0, 8, 8)
        b = build_contour([(2, 2), (6, 2), (6, 8), (2, 8)])
        self.assertEqual(boolean_union(a, b).area, Fraction(64))
        self.assertEqual(
            boolean_intersection(a, b).area, Fraction(24)
        )
        d = boolean_difference(a, b)
        self.assertEqual(d.area, Fraction(40))
        self.assertEqual(len(d.contours), 1)  # no hole: shares the edge

    def test_identical_polygon_reversed_orientation(self):
        s = square(0, 0, 3, 3)
        sr = reverse_contour(s)
        self.assertEqual(boolean_union(s, sr).area, Fraction(9))
        self.assertEqual(boolean_intersection(s, sr).area, Fraction(9))
        self.assertEqual(boolean_difference(s, sr).area, Fraction(0))
        self.assertEqual(len(boolean_union(s, sr).contours), 1)

    def test_output_normalized_when_inputs_are_clockwise(self):
        a = reverse_contour(square(0, 0, 8, 8))
        b = reverse_contour(build_contour(
            [(2, 2), (6, 2), (6, 8), (2, 8)]))
        d = boolean_difference(a, b)
        self.assertEqual(d.area, Fraction(40))
        # Outer loops must still come out CCW regardless of input sense.
        self.assertTrue(all(signed_area(c) > 0 for c in d.contours))


# ---------------------------------------------------------------------------
# Rasterized cross-check against the independent Fraction oracle
# ---------------------------------------------------------------------------


class RasterizationOracleTests(unittest.TestCase):
    def _check(self, a, b, w=12, h=12):
        raw_a, raw_b = contour_to_raw(a), contour_to_raw(b)
        for op, fn in (
            (UNION, boolean_union),
            (INTERSECTION, boolean_intersection),
            (DIFFERENCE, boolean_difference),
        ):
            result = fn(a, b)
            assert_matches_grid(
                self, result.contours, raw_a, raw_b, op, w, h
            )

    def test_hand_built_concave_pair(self):
        lshape = build_contour(
            [(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6)]
        )
        s = square(1, 1, 5, 5)
        self._check(lshape, s)
        self.assertEqual(boolean_union(lshape, s).area, Fraction(29))
        self.assertEqual(boolean_intersection(lshape, s).area, Fraction(7))
        self.assertEqual(boolean_difference(lshape, s).area, Fraction(13))

    def test_offset_triangles(self):
        t1 = build_contour([(1, 1), (9, 1), (1, 9)])
        t2 = build_contour([(3, 0), (11, 4), (3, 8)])
        self._check(t1, t2)

    def test_random_convex_pairs(self):
        rng = random.Random(20260924)
        for case in range(24):
            a = hull_contour(rng, 3, 6, scale=10)
            b = hull_contour(rng, 3, 6, scale=10)
            with self.subTest(case=case):
                self._check(a, b)

    def test_random_pairs_with_translations(self):
        rng = random.Random(777)
        for case in range(12):
            base = hull_contour(rng, 4, 7, scale=6)
            dx = Fraction(rng.randint(-4, 5))
            dy = Fraction(rng.randint(-4, 5))
            shifted = build_contour(
                [
                    (
                        float(Fraction(seg.p0.x) + dx),
                        float(Fraction(seg.p0.y) + dy),
                    )
                    for seg in base.segments
                ]
            )
            with self.subTest(case=case):
                self._check(base, shifted)

    def test_union_raster_equals_two_contour_fill(self):
        # Supplementary identity: under nonzero winding, the union of two
        # CCW polygons fills exactly the same samples as both contours at
        # once.
        a, b = square(0, 0, 5, 5), square(3, 2, 8, 7)
        u = boolean_union(a, b)
        together = rasterize([a, b], 9, 9)
        merged = rasterize(list(u.contours), 9, 9)
        self.assertEqual(together.levels, merged.levels)

    def test_difference_raster_equals_signed_overlap_fill(self):
        # Difference A\B where B is fully inside A: the hole contour
        # traversed CW reproduces the difference under nonzero winding.
        outer = square(0, 0, 8, 8)
        inner = square(2, 2, 6, 6)
        d = boolean_difference(outer, inner)
        via_hole = rasterize([outer, reverse_contour(inner)], 9, 9)
        via_result = rasterize(list(d.contours), 9, 9)
        self.assertEqual(via_hole.levels, via_result.levels)


def _fraction_edges_of(pts_loops):
    return [[(lp[k], lp[(k + 1) % len(lp)]) for k in range(len(lp))]
            for lp in pts_loops]


def _point_on_segment(q, a, b):
    if (b[0] - a[0]) * (q[1] - a[1]) != (b[1] - a[1]) * (q[0] - a[0]):
        return False
    return (
        min(a[0], b[0]) <= q[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= q[1] <= max(a[1], b[1])
    )


def _winding_fraction(edges, qx, qy):
    """Independent exact ray-cast winding (Fraction edges)."""
    total = 0
    for (x0, y0), (x1, y1) in edges:
        if y0 == y1:
            continue
        if y1 > y0:
            lo, hi, d = y0, y1, 1
        else:
            lo, hi, d = y1, y0, -1
        if lo <= qy < hi:
            x = x0 + (qy - y0) * (x1 - x0) / (y1 - y0)
            if x >= qx:
                total += d
    return total


def monotone_polygon(rng):
    """Guaranteed-simple x-monotone polygon: one lower + one upper chain."""
    n = rng.randint(3, 6)
    xs = sorted({Fraction(rng.randint(0, 10), 2) for _ in range(n)})
    while len(xs) < 3:
        xs.append(xs[-1] + 1)
        xs = sorted(xs)
    lower = [(x, Fraction(rng.randint(0, 3))) for x in xs]
    upper = [(x, Fraction(rng.randint(5, 9))) for x in reversed(xs)]
    return lower + upper


class ExactFractionRegionTests(unittest.TestCase):
    """Membership agreement tested entirely in exact Fraction space.

    The boolean core's result loops are compared, point by point on a
    fine lattice, with the independent winding oracle -- no float, no
    rasterizer involved.
    """

    def _check_exact(self, pts_a, pts_b):
        from winding_fill.boolean import _boolean_loops
        a = build_contour([(float(x), float(y)) for x, y in pts_a])
        b = build_contour([(float(x), float(y)) for x, y in pts_b])
        ea = [(pts_a[k], pts_a[(k + 1) % len(pts_a)])
              for k in range(len(pts_a))]
        eb = [(pts_b[k], pts_b[(k + 1) % len(pts_b)])
              for k in range(len(pts_b))]
        for op in (UNION, INTERSECTION, DIFFERENCE):
            loops = _boolean_loops(a, b, op)
            result_edge_sets = _fraction_edges_of(loops)
            for tnum in range(0, 21):
                qx = Fraction(tnum, 2)
                for tny in range(0, 21):
                    qy = Fraction(tny, 2)
                    on_input = any(
                        _point_on_segment((qx, qy), p, q)
                        for es in (ea, eb) for p, q in es
                    )
                    on_result = any(
                        _point_on_segment((qx, qy), p, q)
                        for es in result_edge_sets for p, q in es
                    )
                    wa = _winding_fraction(ea, qx, qy)
                    wb = _winding_fraction(eb, qx, qy)
                    ina, inb = wa != 0, wb != 0
                    expected = (
                        (ina or inb) if op == UNION
                        else (ina and inb) if op == INTERSECTION
                        else (ina and not inb)
                    )
                    wr = sum(
                        _winding_fraction(es, qx, qy)
                        for es in result_edge_sets
                    )
                    got = wr != 0
                    if not on_input and not on_result:
                        self.assertEqual(
                            got, expected,
                            f"{op} at ({qx},{qy}) wa={wa} wb={wb} wr={wr}",
                        )
                    else:
                        # Boundary lattice points: either side may claim
                        # them, but the result can never be INSIDE while
                        # the exact rule puts the point strictly outside.
                        if not expected:
                            self.assertEqual(
                                wr, 0,
                                f"{op}: point ({qx},{qy}) outside inputs "
                                f"but result winding {wr}",
                            )

    def test_hand_pair(self):
        self._check_exact(
            [(0, 0), (4, 0), (4, 4), (0, 4)],
            [(2, 2), (6, 2), (6, 6), (2, 6)],
        )

    def test_concave_pair(self):
        self._check_exact(
            [(Fraction(0), Fraction(0)), (Fraction(6), Fraction(0)),
             (Fraction(6), Fraction(2)), (Fraction(2), Fraction(2)),
             (Fraction(2), Fraction(6)), (Fraction(0), Fraction(6))],
            [(Fraction(1), Fraction(1)), (Fraction(5), Fraction(1)),
             (Fraction(5), Fraction(5)), (Fraction(1), Fraction(5))],
        )

    def test_random_monotone_pairs(self):
        rng = random.Random(31415)
        for case in range(16):
            with self.subTest(case=case):
                self._check_exact(monotone_polygon(rng),
                                  monotone_polygon(rng))


# ---------------------------------------------------------------------------
# Input rejection
# ---------------------------------------------------------------------------


class BooleanValidationTests(unittest.TestCase):
    def setUp(self):
        self.sq = square(0, 0, 4, 4)

    def test_self_intersecting_bowtie_rejected(self):
        bowtie = build_contour([(1, 1), (5, 5), (5, 1), (1, 5)])
        with self.assertRaises(InvalidInputError) as cm:
            boolean_union(bowtie, self.sq)
        self.assertIn("contours[0]", cm.exception.location)
        with self.assertRaises(InvalidInputError) as cm:
            boolean_intersection(self.sq, bowtie)
        self.assertIn("contours[1]", cm.exception.location)

    def test_repeated_vertex_pinch_rejected(self):
        # Two rectangles joined at one corner, expressed as one loop that
        # revisits the joint vertex (4,4).
        pinch = build_contour(
            [(0, 0), (4, 0), (4, 4), (6, 4), (6, 6),
             (4, 6), (4, 4), (0, 4)]
        )
        with self.assertRaises(InvalidInputError) as cm:
            boolean_difference(pinch, self.sq)
        self.assertIn("not a simple polygon", str(cm.exception))

    def test_nonadjacent_edges_touching_rejected(self):
        # Non-adjacent edges (6,6)->(2,4) and (6,2)->(0,2) cross.
        spiky = build_contour(
            [(0, 0), (6, 0), (6, 6), (2, 4), (6, 2), (0, 2)]
        )
        with self.assertRaises(InvalidInputError):
            boolean_union(spiky, self.sq)

    def test_overlapping_adjacent_edges_rejected(self):
        # Backtracking collinear spike: edge (2,0)->(0,0) lies on edge
        # (0,0)->(4,0).
        raw = [
            [
                {"type": "line", "p0": {"x": 0, "y": 0},
                 "p1": {"x": 4, "y": 0}},
                {"type": "line", "p0": {"x": 4, "y": 0},
                 "p1": {"x": 2, "y": 0}},
                {"type": "line", "p0": {"x": 2, "y": 0},
                 "p1": {"x": 0, "y": 0}},
            ]
        ]
        # parse_contours rejects this already (zero net closure aside);
        # build a closed version instead:
        raw = [
            [
                {"type": "line", "p0": {"x": 0, "y": 0},
                 "p1": {"x": 4, "y": 0}},
                {"type": "line", "p0": {"x": 4, "y": 0},
                 "p1": {"x": 4, "y": 4}},
                {"type": "line", "p0": {"x": 4, "y": 4},
                 "p1": {"x": 2, "y": 0}},
                {"type": "line", "p0": {"x": 2, "y": 0},
                 "p1": {"x": 3, "y": 4}},
                {"type": "line", "p0": {"x": 3, "y": 4},
                 "p1": {"x": 0, "y": 4}},
                {"type": "line", "p0": {"x": 0, "y": 4},
                 "p1": {"x": 0, "y": 0}},
            ]
        ]
        from winding_fill import parse_contours
        con = parse_contours(raw)[0]
        with self.assertRaises(InvalidInputError):
            boolean_union(con, self.sq)

    def test_quadratic_contour_rejected(self):
        curved = build_contour(
            [(0, 0), (4, 0), (4, 4), (0, 4)], quads=((0, 2, -1),)
        )
        with self.assertRaises(InvalidInputError) as cm:
            boolean_union(curved, self.sq)
        self.assertIn("quadratic", str(cm.exception))

    def test_non_contour_arguments_rejected(self):
        for bad in (None, 42, "x", [1, 2, 3],
                    [Line(Point(0, 0), Point(1, 0))]):
            with self.assertRaises(InvalidInputError):
                boolean_union(bad, self.sq)
            with self.assertRaises(InvalidInputError):
                boolean_intersection(self.sq, bad)

    def test_raw_entrypoint_validation(self):
        a = [[
            {"type": "line", "p0": {"x": 0, "y": 0},
             "p1": {"x": 4, "y": 0}},
            {"type": "line", "p0": {"x": 4, "y": 0},
             "p1": {"x": 4, "y": 4}},
            {"type": "line", "p0": {"x": 4, "y": 4},
             "p1": {"x": 0, "y": 4}},
            {"type": "line", "p0": {"x": 0, "y": 4},
             "p1": {"x": 0, "y": 0}},
        ]]
        b = [[
            {"type": "line", "p0": {"x": 2, "y": 2},
             "p1": {"x": 6, "y": 2}},
            {"type": "line", "p0": {"x": 6, "y": 2},
             "p1": {"x": 6, "y": 6}},
            {"type": "line", "p0": {"x": 6, "y": 6},
             "p1": {"x": 2, "y": 6}},
            {"type": "line", "p0": {"x": 2, "y": 6},
             "p1": {"x": 2, "y": 2}},
        ]]
        # bare contour and wrapped glyph both accepted
        r1 = boolean_op_raw(a[0], b[0], UNION)
        r2 = boolean_op_raw(a, b, UNION)
        self.assertEqual(r1.area, r2.area)
        self.assertEqual(r1.area, Fraction(28))
        self.assertEqual(r1.op, UNION)
        with self.assertRaises(InvalidInputError):
            boolean_op_raw(a, b, "xor")
        import math
        bad = [[
            {"type": "line", "p0": {"x": 0, "y": 0},
             "p1": {"x": math.nan, "y": 0}},
            {"type": "line", "p0": {"x": math.nan, "y": 0},
             "p1": {"x": 4, "y": 4}},
            {"type": "line", "p0": {"x": 4, "y": 4},
             "p1": {"x": 0, "y": 4}},
            {"type": "line", "p0": {"x": 0, "y": 4},
             "p1": {"x": 0, "y": 0}},
        ]]
        with self.assertRaises(InvalidInputError) as cm:
            boolean_op_raw(bad, b, UNION)
        self.assertIn("NaN", str(cm.exception))

    def test_inputs_not_mutated_on_failure(self):
        bowtie = build_contour([(1, 1), (5, 5), (5, 1), (1, 5)])
        before = tuple(bowtie.segments)
        with self.assertRaises(InvalidInputError):
            boolean_union(bowtie, self.sq)
        self.assertEqual(bowtie.segments, before)
        self.assertEqual(
            tuple((s.p0.x, s.p0.y) for s in self.sq.segments),
            ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)),
        )

    def test_signed_area_helper(self):
        self.assertEqual(signed_area(self.sq), Fraction(16))
        self.assertEqual(signed_area(reverse_contour(self.sq)), Fraction(-16))
        tri = build_contour([(0, 0), (4, 0), (0, 3)])
        self.assertEqual(signed_area(tri), Fraction(6))
        curved = build_contour(
            [(0, 0), (4, 0), (4, 4)], quads=((2, 0, 5),)
        )
        with self.assertRaises(InvalidInputError):
            signed_area(curved)


if __name__ == "__main__":
    unittest.main()
