"""Algorithm tests against an independent reference.

The reference in :mod:`independent_reference` never imports
:mod:`winding_fill`: line cases use exact Fraction arithmetic, curved
cases 60-digit Decimal bisection.  Round-trip style checks (rasterize ->
re-rasterize) are used only as supplementary checks.
"""

from __future__ import annotations

import random
import unittest

from winding_fill import build_contour, rasterize, rasterize_raw

from independent_reference import (
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
