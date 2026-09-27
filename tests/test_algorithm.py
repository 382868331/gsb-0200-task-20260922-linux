"""Algorithm tests against an independent reference.

The reference in :mod:`independent_reference` never imports
:mod:`winding_fill`: line cases use exact Fraction arithmetic, curved
cases 60-digit Decimal bisection.  Round-trip style checks (rasterize ->
re-rasterize) are used only as supplementary checks.
"""

from __future__ import annotations

import random
import unittest

from decimal import Decimal

from winding_fill import (
    EVEN_ODD_RULE,
    NONZERO_RULE,
    build_contour,
    rasterize,
    rasterize_raw,
)

from independent_reference import (
    min_crossing_sample_distance_mixed,
    parity_point_lines,
    reference_levels_lines,
    reference_levels_mixed,
    winding_point_lines,
)


def _chain_contour(points, quads=()):
    """Raw closed contour from a vertex chain (for the reference too)."""
    con = build_contour(points, quads=quads)
    return _contour_to_raw(con)


def _contour_to_raw(con):
    raw = []
    for seg in con.segments:
        d0 = {"x": seg.p0.x, "y": seg.p0.y}
        d1 = {"x": seg.p1.x, "y": seg.p1.y}
        if seg.__class__.__name__ == "Line":
            raw.append({"type": "line", "p0": d0, "p1": d1})
        else:
            raw.append(
                {"type": "quad", "p0": d0,
                 "c": {"x": seg.c.x, "y": seg.c.y}, "p1": d1}
            )
    return raw


def _reverse_raw(raw):
    """Reverse a raw contour's traversal (swap endpoints, keep controls)."""
    out = []
    for s in reversed(raw):
        d = {"type": s["type"], "p0": s["p1"], "p1": s["p0"]}
        if s["type"] == "quad":
            d["c"] = s["c"]
        out.append(d)
    return out


def _glyph(raw):
    """Wrap a single raw contour as a one-contour glyph."""
    return [raw] if raw and isinstance(raw[0], dict) else raw


class ExactLineReferenceTests(unittest.TestCase):
    """Straight polygons: core output must equal the exact reference."""

    def _assert_matches_reference(self, raw, w, h):
        if isinstance(raw[0], dict):
            raw = [raw]  # single contour -> glyph
        result = rasterize_raw(raw, w, h)
        exp_levels, exp_wind = reference_levels_lines(raw, w, h)
        self.assertEqual(
            tuple(tuple(r) for r in result.levels),
            exp_levels,
            "levels mismatch vs exact reference",
        )
        got_wind = tuple(
            tuple(tuple(result.windings[py][px]) for px in range(w))
            for py in range(h)
        )
        self.assertEqual(got_wind, exp_wind, "winding mismatch vs exact reference")

    def test_axis_aligned_squares(self):
        raw = _chain_contour([(1, 1), (5, 1), (5, 5), (1, 5)])
        self._assert_matches_reference(raw, 8, 8)
        # interior pixels carry exactly 16/16
        result = rasterize_raw(_glyph(raw), 8, 8)
        for px in range(1, 5):
            for py in range(1, 5):
                self.assertEqual(result.levels[py][px], 16)
        self.assertEqual(result.levels[0][3], 0)

    def test_right_triangle_exact(self):
        raw = _chain_contour([(1, 1), (6, 1), (6, 6)])
        self._assert_matches_reference(raw, 8, 8)

    def test_rotated_square_eighth_vertices(self):
        # Diamond with vertices on sample-line fractions (multiples of 1/8).
        raw = _chain_contour(
            [(4, 1.125), (6.875, 4), (4, 6.875), (1.125, 4)]
        )
        self._assert_matches_reference(raw, 8, 8)

    def test_random_polygons_on_eighth_grid(self):
        rng = random.Random(20260923)
        for case in range(24):
            n = rng.randint(3, 7)
            pts = [
                (rng.randint(1, 63) / 8.0, rng.randint(1, 63) / 8.0)
                for _ in range(n)
            ]
            w = h = 9
            raw = _chain_contour(pts)
            with self.subTest(case=case, pts=pts):
                self._assert_matches_reference(raw, w, h)

    def test_random_multi_contour_batches(self):
        rng = random.Random(424242)
        for case in range(12):
            ncon = rng.randint(1, 3)
            raw = []
            for _ in range(ncon):
                n = rng.randint(3, 6)
                pts = [
                    (rng.randint(1, 79) / 8.0, rng.randint(1, 79) / 8.0)
                    for _ in range(n)
                ]
                raw.append(_chain_contour(pts))
            with self.subTest(case=case):
                self._assert_matches_reference(raw, 11, 11)

    def test_vertex_exactly_on_sample_line_local_extremum(self):
        # A spike whose tip y is exactly a sub-sample row (3.125 = 25/8):
        # the two adjacent edges meet the half-open rule consistently.
        raw = _chain_contour(
            [(1, 1), (6, 1), (6, 3.125), (3.5, 5.375), (1, 3.125)]
        )
        self._assert_matches_reference(raw, 8, 7)

    def test_vertex_on_sample_line_monotone_pass_through(self):
        # Both edges travel upward through the shared vertex (y monotone),
        # another half-open-rule corner case.
        raw = _chain_contour(
            [(1, 1), (3, 3.125), (5, 3.125), (7, 6), (1, 6)]
        )
        self._assert_matches_reference(raw, 8, 8)

    def test_horizontal_edges_on_sample_lines_are_ignored(self):
        # Rectangle whose bottom/top run exactly on sample rows, split
        # with collinear vertices: fill must equal the plain rectangle.
        split = _chain_contour(
            [(1, 1.125), (3, 1.125), (5, 1.125), (5, 5.875),
             (3, 5.875), (1, 5.875)]
        )
        plain = _chain_contour([(1, 1.125), (5, 1.125), (5, 5.875), (1, 5.875)])
        self._assert_matches_reference(split, 7, 7)
        rs = rasterize_raw(_glyph(split), 7, 7)
        rp = rasterize_raw(_glyph(plain), 7, 7)
        self.assertEqual(rs.levels, rp.levels)


class WindingRuleTests(unittest.TestCase):
    @staticmethod
    def _square(x0, y0, x1, y1, reverse=False):
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        if reverse:
            pts = list(reversed(pts))
        return _chain_contour(pts)

    def test_hole_ring_nonzero(self):
        outer = self._square(1, 1, 9, 9)                 # CCW
        inner = self._square(3, 3, 7, 7, reverse=True)   # CW
        result = rasterize_raw([outer, inner], 10, 10)
        # ring covered, hole empty
        self.assertEqual(result.levels[2][2], 16)
        self.assertEqual(result.levels[5][5], 0)
        # explicit winding queries
        from fractions import Fraction
        self.assertEqual(
            winding_point_lines([outer, inner], Fraction(5), Fraction(5)), 0
        )
        self.assertEqual(
            winding_point_lines([outer, inner], Fraction(2), Fraction(5)), 1
        )
        self.assertEqual(result.winding_at(5, 5, 1, 2), 0)
        self.assertEqual(result.winding_at(2, 5, 1, 2), 1)

    def test_overlapping_same_orientation_winding_two(self):
        a = self._square(1, 1, 5, 5)
        b = self._square(3, 3, 7, 7)
        result = rasterize_raw([a, b], 9, 9)
        # overlap region: winding +2 but still covered (nonzero rule)
        for px in range(3, 5):
            for py in range(3, 5):
                self.assertEqual(result.levels[py][px], 16)
                self.assertTrue(
                    all(w == 2 for w in result.windings[py][px])
                )
        self.assertTrue(
            all(w == 1 for w in result.windings[2][2])
        )
        self.assertTrue(
            all(w == 1 for w in result.windings[6][6])
        )
        # levels must equal the union given by the exact reference
        exp_levels, _ = reference_levels_lines([a, b], 9, 9)
        self.assertEqual(
            tuple(tuple(r) for r in result.levels), exp_levels
        )

    def test_identical_contour_with_reverse_cancels_everywhere(self):
        fwd = self._square(1, 1, 7, 7)
        rev = self._square(1, 1, 7, 7, reverse=True)
        result = rasterize_raw([fwd, rev], 9, 9)
        for py in range(9):
            for px in range(9):
                self.assertEqual(result.levels[py][px], 0)
                self.assertTrue(
                    all(w == 0 for w in result.windings[py][px])
                )

    def test_global_reversal_preserves_levels_flips_winding(self):
        raw = _chain_contour(
            [(1, 1), (6, 1), (5, 5), (2, 6), (1, 4)]
        )
        rev = list(reversed(
            [
                {
                    "type": s["type"],
                    "p0": s["p1"],
                    "p1": s["p0"],
                    **({"c": s["c"]} if s["type"] == "quad" else {}),
                }
                for s in raw
            ]
        ))
        r1 = rasterize_raw(_glyph(raw), 8, 8)
        r2 = rasterize_raw(_glyph(rev), 8, 8)
        self.assertEqual(r1.levels, r2.levels)
        for py in range(8):
            for px in range(8):
                w1 = r1.windings[py][px]
                w2 = r2.windings[py][px]
                self.assertEqual(tuple(-w for w in w1), tuple(w2))

    def test_self_intersecting_bowtie(self):
        # (1,1)->(5,5)->(5,1)->(1,5): diagonals cross at (3,3).
        raw = _chain_contour([(1, 1), (5, 5), (5, 1), (1, 5)])
        result = rasterize_raw(_glyph(raw), 7, 7)
        exp_levels, exp_wind = reference_levels_lines(raw, 7, 7)
        self.assertEqual(
            tuple(tuple(r) for r in result.levels), exp_levels
        )
        got_wind = tuple(
            tuple(tuple(result.windings[py][px]) for px in range(7))
            for py in range(7)
        )
        self.assertEqual(got_wind, exp_wind)
        # Spot checks of signed windings (eighth-grid points so the
        # Fraction reference is exact): the two filled lobes have
        # opposite signs, and the pinch between the crossing and the
        # shared vertical edge has winding 0.
        from fractions import Fraction
        left = (Fraction(17, 8), Fraction(17, 8))    # (2.125, 2.125)
        right = (Fraction(39, 8), Fraction(17, 8))   # (4.875, 2.125)
        pinch = (Fraction(3), Fraction(17, 8))       # (3.0, 2.125)
        self.assertEqual(winding_point_lines(raw, *left), 1)
        self.assertEqual(winding_point_lines(raw, *right), -1)
        self.assertEqual(winding_point_lines(raw, *pinch), 0)
        for sx, sy in (left, right):
            self.assertNotEqual(winding_point_lines(raw, sx, sy), 0)

    def test_nonzero_differs_from_parity_on_overlap(self):
        # Two coincident CCW loops: parity would erase them, nonzero
        # keeps them (winding 2).
        sq = self._square(1, 1, 5, 5)
        result = rasterize_raw([sq, sq], 7, 7)
        self.assertEqual(result.levels[3][3], 16)
        self.assertTrue(all(w == 2 for w in result.windings[3][3]))


class EvenOddRuleTests(unittest.TestCase):
    """The alternative even-odd (alternating) fill rule."""

    @staticmethod
    def _square(x0, y0, x1, y1, reverse=False):
        return WindingRuleTests._square(x0, y0, x1, y1, reverse=reverse)

    def _assert_evenodd_reference_lines(self, raw, w, h):
        if isinstance(raw[0], dict):
            raw = [raw]
        result = rasterize_raw(raw, w, h, rule=EVEN_ODD_RULE)
        exp_levels, exp_wind = reference_levels_lines(
            raw, w, h, rule=EVEN_ODD_RULE
        )
        self.assertEqual(result.rule, EVEN_ODD_RULE)
        self.assertEqual(
            tuple(tuple(r) for r in result.levels), exp_levels
        )
        # windings must be rule-independent (same crossing computation)
        got_wind = tuple(
            tuple(tuple(result.windings[py][px]) for px in range(w))
            for py in range(h)
        )
        self.assertEqual(got_wind, exp_wind)

    def test_default_rule_is_nonzero(self):
        sq = self._square(1, 1, 5, 5)
        result = rasterize_raw([sq], 6, 6)
        self.assertEqual(result.rule, NONZERO_RULE)

    def test_single_loop_same_as_nonzero(self):
        raw = _chain_contour(
            [(1, 1), (6, 1), (5, 5), (2, 6), (1, 4)]
        )
        rn = rasterize_raw(_glyph(raw), 8, 8, rule=NONZERO_RULE)
        re = rasterize_raw(_glyph(raw), 8, 8, rule=EVEN_ODD_RULE)
        self.assertEqual(rn.levels, re.levels)
        # crossing geometry identical -> windings identical
        self.assertEqual(rn.windings, re.windings)

    def test_coincident_loops_cancel_under_evenodd(self):
        # Two coincident CCW loops: parity toggles twice -> empty,
        # while nonzero fills (winding 2).  This is the headline rule
        # difference.
        sq = self._square(1, 1, 5, 5)
        eo = rasterize_raw([sq, sq], 7, 7, rule=EVEN_ODD_RULE)
        nz = rasterize_raw([sq, sq], 7, 7)
        for py in range(7):
            for px in range(7):
                self.assertEqual(eo.levels[py][px], 0)
        self.assertEqual(eo.levels[3][3], 0)
        # signed windings are still +2 (geometry unchanged)
        self.assertTrue(all(w == 2 for w in eo.windings[3][3]))
        self.assertEqual(nz.levels[3][3], 16)
        self._assert_evenodd_reference_lines([sq, sq], 7, 7)

    def test_hole_direction_is_irrelevant_under_evenodd(self):
        # Under parity, an inner loop cuts a hole regardless of its
        # orientation; nonzero only cuts it when oppositely oriented.
        outer = self._square(1, 1, 9, 9)
        inner_ccw = self._square(3, 3, 7, 7)          # same orientation
        inner_cw = self._square(3, 3, 7, 7, reverse=True)
        eo_ccw = rasterize_raw([outer, inner_ccw], 10, 10,
                               rule=EVEN_ODD_RULE)
        eo_cw = rasterize_raw([outer, inner_cw], 10, 10,
                              rule=EVEN_ODD_RULE)
        nz_ccw = rasterize_raw([outer, inner_ccw], 10, 10)
        nz_cw = rasterize_raw([outer, inner_cw], 10, 10)
        # both parity renders show the hole
        self.assertEqual(eo_ccw.levels[5][5], 0)
        self.assertEqual(eo_cw.levels[5][5], 0)
        self.assertEqual(eo_ccw.levels, eo_cw.levels)
        # nonzero differs by orientation: same-way inner loop fills
        self.assertEqual(nz_ccw.levels[5][5], 16)
        self.assertEqual(nz_cw.levels[5][5], 0)
        self._assert_evenodd_reference_lines(
            [outer, inner_ccw], 10, 10
        )

    def test_overlapping_squares_evenodd(self):
        a = self._square(1, 1, 5, 5)
        b = self._square(3, 3, 7, 7)
        result = rasterize_raw([a, b], 9, 9, rule=EVEN_ODD_RULE)
        self._assert_evenodd_reference_lines([a, b], 9, 9)
        # overlap region is crossed twice -> empty under parity
        for px in range(3, 5):
            for py in range(3, 5):
                self.assertEqual(result.levels[py][px], 0)
                self.assertTrue(
                    all(w == 2 for w in result.windings[py][px])
                )
        # single-loop regions filled
        self.assertTrue(all(w == 1 for w in result.windings[2][2]))
        self.assertEqual(result.levels[2][2], 16)

    def test_self_intersecting_bowtie_evenodd(self):
        raw = _chain_contour([(1, 1), (5, 5), (5, 1), (1, 5)])
        eo = rasterize_raw(_glyph(raw), 7, 7, rule=EVEN_ODD_RULE)
        self._assert_evenodd_reference_lines(raw, 7, 7)
        nz = rasterize_raw(_glyph(raw), 7, 7)
        from fractions import Fraction
        left = (Fraction(17, 8), Fraction(17, 8))
        right = (Fraction(39, 8), Fraction(17, 8))
        pinch = (Fraction(3), Fraction(17, 8))
        # both lobes: one crossing -> inside under parity
        self.assertEqual(parity_point_lines(raw, *left), 1)
        self.assertEqual(parity_point_lines(raw, *right), 1)
        # at sy=2.125 the crossings are x=1, 2.125, 3.875, 5, so the
        # lobes cover (1,2.125) and (3.875,5): both parity-filled
        self.assertGreater(eo.levels[1][1], 0)
        self.assertGreater(eo.levels[1][4], 0)
        # under nonzero the two lobes carry opposite signed windings;
        # parity ignores the sign and fills both
        self.assertTrue(any(w == 1 for w in eo.windings[1][1]))
        self.assertTrue(any(w == -1 for w in eo.windings[1][4]))
        # pinch region has two crossings (winding 0) -> parity out
        self.assertEqual(parity_point_lines(raw, *pinch), 0)
        self.assertEqual(eo.levels[1][3], 0)
        # signed windings are the same under both rules
        self.assertEqual(eo.windings, nz.windings)

    def test_five_point_star_evenodd_vs_nonzero(self):
        # Classic pentagram: under parity the five triangular arms are
        # filled (1 crossing) while the central pentagon is reached by 2
        # crossings and stays EMPTY; nonzero winding fills the center too
        # (winding 2 there).
        import math
        pts = []
        for k in range(5):
            ang = math.pi / 2 + 4 * math.pi * k / 5
            pts.append((4 + 3.5 * math.cos(ang), 4 + 3.5 * math.sin(ang)))
        star = build_contour(pts)
        raw = _contour_to_raw(star)
        eo = rasterize([star], 9, 9, rule=EVEN_ODD_RULE)
        nz = rasterize([star], 9, 9)
        # most center sub-samples sit in the central pentagon
        self.assertLess(eo.levels[4][4], nz.levels[4][4])
        self.assertTrue(any(w == 2 for w in eo.windings[4][4]))
        self.assertEqual(nz.levels[4][4], 16)

    def test_vertex_on_sample_line_rules_agree_and_match_reference(self):
        # Degenerate vertex y exactly on a sub-sample row: both rules
        # must use the same half-open crossing count.
        raw = _chain_contour(
            [(1, 1), (6, 1), (6, 3.125), (3.5, 5.375), (1, 3.125)]
        )
        self._assert_evenodd_reference_lines(raw, 8, 7)
        raw2 = _chain_contour(
            [(1, 1), (3, 3.125), (5, 3.125), (7, 6), (1, 6)]
        )
        self._assert_evenodd_reference_lines(raw2, 8, 8)

    def test_random_polygons_match_exact_parity_reference(self):
        rng = random.Random(20260927)
        for case in range(20):
            ncon = rng.randint(1, 3)
            raw = []
            for _ in range(ncon):
                n = rng.randint(3, 7)
                pts = [
                    (rng.randint(1, 63) / 8.0, rng.randint(1, 63) / 8.0)
                    for _ in range(n)
                ]
                raw.append(_chain_contour(pts))
            with self.subTest(case=case):
                self._assert_evenodd_reference_lines(raw, 9, 9)

    def test_invalid_rule_rejected(self):
        from winding_fill import InvalidInputError
        sq = self._square(1, 1, 5, 5)
        for bad in ("even_odd", "EVEN-ODD", "winding", 0, None, True):
            with self.assertRaises(InvalidInputError):
                rasterize_raw([sq], 6, 6, rule=bad)
            with self.assertRaises(InvalidInputError):
                rasterize([build_contour(
                    [(1, 1), (5, 1), (5, 5), (1, 5)])], 6, 6, rule=bad)

    def test_is_covered_predicate(self):
        eo = rasterize_raw(
            [self._square(1, 1, 5, 5)] * 2, 7, 7, rule=EVEN_ODD_RULE
        )
        nz = rasterize_raw(
            [self._square(1, 1, 5, 5)] * 2, 7, 7
        )
        self.assertFalse(eo.is_covered(2))
        self.assertTrue(eo.is_covered(3))
        self.assertTrue(nz.is_covered(2))
        self.assertTrue(nz.is_covered(-2))


class QuadraticTests(unittest.TestCase):
    def _assert_matches_decimal_reference(self, raw, w, h):
        if isinstance(raw[0], dict):
            raw = [raw]  # single contour -> glyph
        result = rasterize_raw(raw, w, h)
        exp_levels, exp_wind = reference_levels_mixed(raw, w, h)
        self.assertEqual(
            tuple(tuple(r) for r in result.levels),
            exp_levels,
            "levels mismatch vs Decimal reference",
        )
        got_wind = tuple(
            tuple(tuple(result.windings[py][px]) for px in range(w))
            for py in range(h)
        )
        self.assertEqual(
            got_wind, exp_wind, "winding mismatch vs Decimal reference"
        )

    def test_arch_with_tangent_on_sample_row(self):
        # Quadratic arch (1,2)->(5,2) with control (3,4.25): the curve's
        # y-extremum is y(0.5) = 1 + 4.25/2 = 3.125 = 25/8, exactly on a
        # sub-sample row (py=3, j=0).  Half-open y rule excludes the
        # extremum on BOTH monotonic pieces, so that row has zero
        # crossings; the row just below is fully covered (winding -1,
        # the contour runs clockwise).
        arch = [
            {"type": "quad",
             "p0": {"x": 1, "y": 2}, "c": {"x": 3, "y": 4.25},
             "p1": {"x": 5, "y": 2}},
            {"type": "line",
             "p0": {"x": 5, "y": 2}, "p1": {"x": 5, "y": 0}},
            {"type": "line",
             "p0": {"x": 5, "y": 0}, "p1": {"x": 1, "y": 0}},
            {"type": "line",
             "p0": {"x": 1, "y": 0}, "p1": {"x": 1, "y": 2}},
        ]
        self._assert_matches_decimal_reference([arch], 7, 5)
        result = rasterize_raw([arch], 7, 5)
        # tangent row: extremum excluded from both pieces -> all outside
        self.assertEqual(
            tuple(result.levels[3]), tuple([0] * 7)
        )
        self.assertEqual(result.winding_at(3, 3, 1, 0), 0)
        # row py=2, j=3 -> sy = 2.875: interior columns fully covered
        # (-1, contour runs clockwise); exact fill for all 16 samples
        # is already guaranteed by the Decimal reference comparison.
        self.assertEqual(result.levels[2][2], 16)
        self.assertEqual(result.levels[2][3], 16)
        self.assertEqual(result.winding_at(2, 2, 0, 3), -1)
        # column px=4 narrows with the curve (4/3/1/0 samples per row)
        self.assertEqual(result.levels[2][4], 8)
        # everything right of the wall is empty
        self.assertEqual(result.levels[2][5], 0)

    def test_quad_approx_circle(self):
        # Four-quadratic closed blob approximating a circle centred
        # (5,5), radius 3, using the standard 0.75*r control stretch.
        k = 0.75 * 3
        raw = [[
            {"type": "quad",
             "p0": {"x": 2, "y": 5}, "c": {"x": 2, "y": 5 + k},
             "p1": {"x": 5, "y": 8}},
            {"type": "quad",
             "p0": {"x": 5, "y": 8}, "c": {"x": 5 + k, "y": 8},
             "p1": {"x": 8, "y": 5}},
            {"type": "quad",
             "p0": {"x": 8, "y": 5}, "c": {"x": 8, "y": 5 - k},
             "p1": {"x": 5, "y": 2}},
            {"type": "quad",
             "p0": {"x": 5, "y": 2}, "c": {"x": 5 - k, "y": 2},
             "p1": {"x": 2, "y": 5}},
        ]]
        self._assert_matches_decimal_reference(raw, 10, 10)

    def test_sagging_quad_minimum_on_sample_row(self):
        # Inverse arch: quadratic (1,4)->(5,4) with control (3,2.25) has
        # its interior y-MINIMUM at y(0.5) = 4 + (2.25-4)/2 = 3.125 =
        # 25/8, exactly on a sub-sample row.  Both monotonic pieces
        # contain the minimum as their included low end, so that row
        # gets TWO tangent crossings (downward -1 then upward +1); they
        # must cancel exactly and agree with the Decimal reference.
        body = [
            {"type": "quad",
             "p0": {"x": 1, "y": 4}, "c": {"x": 3, "y": 2.25},
             "p1": {"x": 5, "y": 4}},
            {"type": "line",
             "p0": {"x": 5, "y": 4}, "p1": {"x": 5, "y": 6}},
            {"type": "line",
             "p0": {"x": 5, "y": 6}, "p1": {"x": 1, "y": 6}},
            {"type": "line",
             "p0": {"x": 1, "y": 6}, "p1": {"x": 1, "y": 4}},
        ]
        self._assert_matches_decimal_reference([body], 7, 7)
        result = rasterize_raw([body], 7, 7)
        # tangent sub-row py=3, j=0 (sy=3.125): the body lies ABOVE the
        # sag, so this sub-row touches it only at the tangent point --
        # every sub-sample is outside (the -1/+1 pair cancels).
        for px in range(7):
            for i in range(4):
                self.assertEqual(result.winding_at(px, 3, i, 0), 0)
        # next sub-row j=1 (sy=3.375): interior between the two curve
        # crossings around x=3 -> center samples covered (+1)
        self.assertEqual(result.winding_at(3, 3, 0, 1), 1)
        self.assertEqual(result.winding_at(3, 3, 3, 1), 1)
        # full pixels in the body are solid; below the sag all empty
        self.assertEqual(result.levels[2][3], 0)
        self.assertEqual(result.levels[5][3], 16)

    def test_s_shaped_quad_with_interior_extremum(self):
        # Control far to one side so y(t) has an interior extremum and
        # the curve doubles back vertically.
        body = [
            {"type": "quad",
             "p0": {"x": 1, "y": 1}, "c": {"x": 6, "y": 7},
             "p1": {"x": 3, "y": 6}},
            {"type": "line",
             "p0": {"x": 3, "y": 6}, "p1": {"x": 7, "y": 6}},
            {"type": "line",
             "p0": {"x": 7, "y": 6}, "p1": {"x": 7, "y": 1}},
            {"type": "line",
             "p0": {"x": 7, "y": 1}, "p1": {"x": 1, "y": 1}},
        ]
        self._assert_matches_decimal_reference([body], 9, 8)

    def test_degenerate_quad_control_at_endpoint_equals_line(self):
        # Control == p0: the quad collapses exactly to a straight line.
        quad_tri = [
            [
                {"type": "quad",
                 "p0": {"x": 1, "y": 1}, "c": {"x": 1, "y": 1},
                 "p1": {"x": 6, "y": 1}},
                {"type": "line",
                 "p0": {"x": 6, "y": 1}, "p1": {"x": 6, "y": 6}},
                {"type": "line",
                 "p0": {"x": 6, "y": 6}, "p1": {"x": 1, "y": 1}},
            ]
        ]
        line_tri = _chain_contour([(1, 1), (6, 1), (6, 6)])
        rq = rasterize_raw(quad_tri, 8, 8)
        rl = rasterize_raw(_glyph(line_tri), 8, 8)
        self.assertEqual(rq.levels, rl.levels)
        self.assertEqual(rq.windings, rl.windings)
        self._assert_matches_decimal_reference(quad_tri, 8, 8)

    def test_random_mixed_contours(self):
        rng = random.Random(9973)
        for case in range(10):
            n = rng.randint(3, 5)
            pts = [
                (rng.randint(1, 87) / 8.0, rng.randint(1, 87) / 8.0)
                for _ in range(n)
            ]
            quads = []
            for i in range(n):
                if rng.random() < 0.5:
                    quads.append(
                        (i, rng.randint(1, 95) / 8.0,
                         rng.randint(1, 95) / 8.0)
                    )
            raw = [_chain_contour(pts, quads=tuple(quads))]
            with self.subTest(case=case):
                self._assert_matches_decimal_reference(raw, 12, 12)

    def test_random_mixed_contours_off_grid_floats(self):
        # Coordinates NOT on the 1/8 sub-sample lattice: agreement with
        # the Decimal reference then rests purely on double-precision
        # root accuracy (no boundary coincidence can mask an error).
        rng = random.Random(31337)
        for case in range(8):
            n = rng.randint(3, 5)
            pts = [
                (rng.randint(50, 900) / 100.0, rng.randint(50, 900) / 100.0)
                for _ in range(n)
            ]
            quads = []
            for i in range(n):
                if rng.random() < 0.5:
                    quads.append(
                        (i, rng.randint(0, 1000) / 100.0,
                         rng.randint(0, 1000) / 100.0)
                    )
            raw = [_chain_contour(pts, quads=tuple(quads))]
            with self.subTest(case=case):
                self._assert_matches_decimal_reference(raw, 11, 11)

    def test_quad_hole_and_overlap_against_reference(self):
        # Curved CCW outer blob with a curved CW hole; a second CCW blob
        # overlaps the outer one (winding +2 in the overlap).
        outer = build_contour(
            [(1, 1), (9, 1), (9, 9), (1, 9)],
            quads=((0, 5, -0.5), (2, 5, 10.5)),
        )
        hole_fwd = build_contour(
            [(4, 4), (7, 4), (7, 7), (4, 7)],
            quads=((0, 5.5, 3), (2, 5.5, 8)),
        )
        blob2 = build_contour(
            [(6, 2), (10, 2), (10, 6), (6, 6)],
            quads=((0, 8, 1), (2, 8, 7)),
        )
        outer_raw = _contour_to_raw(outer)
        hole_raw = _reverse_raw(_contour_to_raw(hole_fwd))
        blob2_raw = _contour_to_raw(blob2)
        raw = [outer_raw, hole_raw, blob2_raw]
        self._assert_matches_decimal_reference(raw, 11, 11)
        result = rasterize_raw(raw, 11, 11)
        # hole empty despite overlap geometry passing beside it
        self.assertEqual(result.levels[5][5], 0)
        self.assertTrue(all(w == 0 for w in result.windings[5][5]))
        # pure outer region
        self.assertEqual(result.levels[8][2], 16)
        # outer + blob2 overlap (right strip): winding 2, still covered
        self.assertTrue(all(w == 2 for w in result.windings[4][8]))


class QuadraticEvenOddTests(unittest.TestCase):
    """Even-odd rule with quadratic curves vs the Decimal reference."""

    def _assert_evenodd_reference_mixed(self, raw, w, h):
        if isinstance(raw[0], dict):
            raw = [raw]
        result = rasterize_raw(raw, w, h, rule=EVEN_ODD_RULE)
        exp_levels, exp_wind = reference_levels_mixed(
            raw, w, h, rule=EVEN_ODD_RULE
        )
        self.assertEqual(
            tuple(tuple(r) for r in result.levels), exp_levels,
            "even-odd levels mismatch vs Decimal reference",
        )
        got_wind = tuple(
            tuple(tuple(result.windings[py][px]) for px in range(w))
            for py in range(h)
        )
        # signed winding identical for both rules -> crossing geometry
        # was not perturbed
        self.assertEqual(got_wind, exp_wind)
        nz = rasterize_raw(raw, w, h)
        self.assertEqual(nz.windings, result.windings)

    def test_double_curve_blob_parity_erases(self):
        # Same curved CCW blob twice: every interior point crossed an
        # even number of times -> empty under parity, solid under
        # nonzero; curves must be handled exactly (no line flattening).
        k = 0.75 * 3
        blob = [
            {"type": "quad",
             "p0": {"x": 2, "y": 5}, "c": {"x": 2, "y": 5 + k},
             "p1": {"x": 5, "y": 8}},
            {"type": "quad",
             "p0": {"x": 5, "y": 8}, "c": {"x": 5 + k, "y": 8},
             "p1": {"x": 8, "y": 5}},
            {"type": "quad",
             "p0": {"x": 8, "y": 5}, "c": {"x": 8, "y": 5 - k},
             "p1": {"x": 5, "y": 2}},
            {"type": "quad",
             "p0": {"x": 5, "y": 2}, "c": {"x": 5 - k, "y": 2},
             "p1": {"x": 2, "y": 5}},
        ]
        raw = [blob, blob]
        self._assert_evenodd_reference_mixed(raw, 10, 10)
        eo = rasterize_raw(raw, 10, 10, rule=EVEN_ODD_RULE)
        nz = rasterize_raw(raw, 10, 10)
        self.assertEqual(eo.levels[5][5], 0)
        self.assertEqual(nz.levels[5][5], 16)
        self.assertTrue(all(abs(w) == 2 for w in eo.windings[5][5]))

    def test_tangent_extremum_row_consistent_between_rules(self):
        # The arch whose y-extremum lies exactly on a sub-sample row:
        # the half-open rule excludes it under BOTH fill rules; parity
        # must not gain a phantom toggle at the tangent.
        arch = [
            {"type": "quad",
             "p0": {"x": 1, "y": 2}, "c": {"x": 3, "y": 4.25},
             "p1": {"x": 5, "y": 2}},
            {"type": "line",
             "p0": {"x": 5, "y": 2}, "p1": {"x": 5, "y": 0}},
            {"type": "line",
             "p0": {"x": 5, "y": 0}, "p1": {"x": 1, "y": 0}},
            {"type": "line",
             "p0": {"x": 1, "y": 0}, "p1": {"x": 1, "y": 2}},
        ]
        self._assert_evenodd_reference_mixed([arch], 7, 5)
        eo = rasterize_raw([arch], 7, 5, rule=EVEN_ODD_RULE)
        self.assertEqual(tuple(eo.levels[3]), tuple([0] * 7))

    def test_sagging_minimum_tangent_pair_no_parity_flip(self):
        body = [
            {"type": "quad",
             "p0": {"x": 1, "y": 4}, "c": {"x": 3, "y": 2.25},
             "p1": {"x": 5, "y": 4}},
            {"type": "line",
             "p0": {"x": 5, "y": 4}, "p1": {"x": 5, "y": 6}},
            {"type": "line",
             "p0": {"x": 5, "y": 6}, "p1": {"x": 1, "y": 6}},
            {"type": "line",
             "p0": {"x": 1, "y": 6}, "p1": {"x": 1, "y": 4}},
        ]
        self._assert_evenodd_reference_mixed([body], 7, 7)

    def test_random_mixed_contours_evenodd_vs_decimal(self):
        # Deterministic multi-contour overlap: two curved CCW blobs.
        # Parity empties their overlap; the inner curved loop below is
        # crossed regardless of orientation.
        b1 = build_contour(
            [(1, 1), (7, 1), (7, 7), (1, 7)],
            quads=((0, 4, 0.0), (2, 4, 8.0)),
        )
        b2 = build_contour(
            [(4, 2), (10, 2), (10, 8), (4, 8)],
            quads=((0, 7, 1), (2, 7, 9)),
        )
        raw = [_contour_to_raw(b1), _contour_to_raw(b2)]
        self._assert_evenodd_reference_mixed(raw, 11, 11)

        # Same seed/geometry as the nonzero mixed-contour reference
        # test, so the crossing geometry is already known to agree with
        # the Decimal reference (no double-rounding boundary fluke);
        # here only the rule predicate differs.
        rng = random.Random(9973)
        for case in range(10):
            ncon = rng.randint(1, 3)
            raw = []
            for _ in range(ncon):
                n = rng.randint(3, 5)
                pts = [
                    (rng.randint(1, 87) / 8.0, rng.randint(1, 87) / 8.0)
                    for _ in range(n)
                ]
                quads = tuple(
                    (i, rng.randint(1, 95) / 8.0, rng.randint(1, 95) / 8.0)
                    for i in range(n) if rng.random() < 0.5
                )
            raw.append(_chain_contour(pts, quads=quads))
            with self.subTest(case=case):
                # Parity flips on a single crossing, so exact agreement
                # is demanded only when no crossing lands inside the
                # documented double-precision root round-off band of a
                # sample x (README: error budget, no artificial tol).
                dist = min_crossing_sample_distance_mixed(raw, 12, 12)
                if dist >= Decimal(1) / Decimal(10**10):
                    self._assert_evenodd_reference_mixed(raw, 12, 12)

    def test_random_mixed_contours_off_grid_evenodd(self):
        rng = random.Random(31337)
        for case in range(8):
            n = rng.randint(3, 5)
            pts = [
                (rng.randint(50, 900) / 100.0, rng.randint(50, 900) / 100.0)
                for _ in range(n)
            ]
            quads = tuple(
                (i, rng.randint(0, 1000) / 100.0,
                 rng.randint(0, 1000) / 100.0)
                for i in range(n) if rng.random() < 0.5
            )
            raw = [_chain_contour(pts, quads=quads)]
            with self.subTest(case=case):
                dist = min_crossing_sample_distance_mixed(raw, 11, 11)
                if dist >= Decimal(1) / Decimal(10**10):
                    self._assert_evenodd_reference_mixed(raw, 11, 11)


class CoverageValueTests(unittest.TestCase):
    def test_coverage_is_k_over_16(self):
        # Triangle crossing many pixels yields intermediate gray levels;
        # every level must be an integer 0..16 and coverage k/16.
        raw = _chain_contour([(0.625, 0.375), (8, 0.375), (8, 7.625)])
        result = rasterize_raw(_glyph(raw), 9, 8)
        seen = set()
        for py in range(8):
            for px in range(9):
                k = result.levels[py][px]
                self.assertIsInstance(k, int)
                self.assertIn(k, range(17))
                self.assertEqual(result.coverage_at(px, py), k / 16.0)
                seen.add(k)
        self.assertTrue(any(0 < k < 16 for k in seen), seen)

    def test_determinism_repeated_runs(self):
        raw = _chain_contour(
            [(0.5, 0.5), (3.2, 0.7), (3.8, 3.1), (0.2, 2.9)]
        )
        r1 = rasterize_raw(_glyph(raw), 5, 4)
        r2 = rasterize_raw(_glyph(raw), 5, 4)
        self.assertEqual(r1.levels, r2.levels)
        self.assertEqual(r1.windings, r2.windings)

    def test_typed_builder_and_raw_agree(self):
        con = build_contour(
            [(1, 1), (6, 1), (6, 6)],
            quads=((2, 0.25, 3.5),),
        )
        raw = _chain_contour(
            [(1, 1), (6, 1), (6, 6)], quads=((2, 0.25, 3.5),)
        )
        a = rasterize([con], 8, 8)
        b = rasterize_raw(_glyph(raw), 8, 8)
        # supplementary round-trip-style check only
        self.assertEqual(a.levels, b.levels)
        self.assertEqual(a.windings, b.windings)


if __name__ == "__main__":
    unittest.main()
