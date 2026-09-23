"""Validation and error-semantics tests."""

from __future__ import annotations

import math
import unittest

from winding_fill import (
    MAX_CANVAS,
    MAX_CONTOURS,
    InvalidInputError,
    build_contour,
    parse_contours,
    rasterize,
    rasterize_raw,
)


def square_raw(x0=1.0, y0=1.0, x1=5.0, y1=5.0):
    return [
        [
            {"type": "line", "p0": {"x": x0, "y": y0}, "p1": {"x": x1, "y": y0}},
            {"type": "line", "p0": {"x": x1, "y": y0}, "p1": {"x": x1, "y": y1}},
            {"type": "line", "p0": {"x": x1, "y": y1}, "p1": {"x": x0, "y": y1}},
            {"type": "line", "p0": {"x": x0, "y": y1}, "p1": {"x": x0, "y": y0}},
        ]
    ]


class NumericInputTests(unittest.TestCase):
    def test_nan_rejected_with_location(self):
        raw = square_raw()
        raw[0][1]["p1"]["y"] = math.nan
        with self.assertRaises(InvalidInputError) as cm:
            parse_contours(raw)
        self.assertIn("contours[0].segments[1].p1.y", cm.exception.location)
        self.assertIn("NaN", str(cm.exception))

    def test_infinity_rejected(self):
        for bad in (math.inf, -math.inf):
            raw = square_raw()
            raw[0][0]["p0"]["x"] = bad
            with self.assertRaises(InvalidInputError) as cm:
                parse_contours(raw)
            self.assertIn("contours[0].segments[0].p0.x", cm.exception.location)

    def test_bool_coordinate_rejected(self):
        raw = square_raw()
        raw[0][0]["p0"]["x"] = True  # bool must not be accepted as int
        with self.assertRaises(InvalidInputError):
            parse_contours(raw)

    def test_string_coordinate_rejected(self):
        raw = square_raw()
        raw[0][0]["p0"]["x"] = "1"
        with self.assertRaises(InvalidInputError):
            parse_contours(raw)

    def test_coordinate_out_of_limit(self):
        with self.assertRaises(InvalidInputError):
            parse_contours(square_raw(x1=1025.0))
        with self.assertRaises(InvalidInputError):
            parse_contours(square_raw(y0=-1025.0))

    def test_nan_canvas_rejected(self):
        # width must be a plain int; floats (incl. NaN) are refused too
        with self.assertRaises(InvalidInputError):
            rasterize_raw(square_raw(), 8.0, 8)

    def test_bool_dimension_rejected(self):
        for w in (True, False):
            with self.assertRaises(InvalidInputError):
                rasterize_raw(square_raw(), w, 8)


class StructuralInputTests(unittest.TestCase):
    def test_contour_count_bounds(self):
        with self.assertRaises(InvalidInputError):
            rasterize_raw([], 8, 8)
        many = square_raw() * (MAX_CONTOURS + 1)
        self.assertEqual(len(many), MAX_CONTOURS + 1)
        with self.assertRaises(InvalidInputError):
            rasterize_raw(many, 8, 8)
        # exactly the limit is accepted
        ok = square_raw() * MAX_CONTOURS
        result = rasterize_raw(ok, 4, 4)
        self.assertEqual(result.coverage_at(2, 2), 1.0)

    def test_canvas_bounds(self):
        with self.assertRaises(InvalidInputError):
            rasterize_raw(square_raw(), 0, 8)
        with self.assertRaises(InvalidInputError):
            rasterize_raw(square_raw(), 8, -1)
        with self.assertRaises(InvalidInputError):
            rasterize_raw(square_raw(), MAX_CANVAS + 1, 8)
        result = rasterize_raw(square_raw(0, 0, 64, 64), MAX_CANVAS, 1)
        self.assertEqual(result.width, MAX_CANVAS)

    def test_too_few_segments(self):
        raw = [
            [
                {"type": "line", "p0": {"x": 0, "y": 0}, "p1": {"x": 1, "y": 0}},
                {"type": "line", "p0": {"x": 1, "y": 0}, "p1": {"x": 0, "y": 0}},
            ]
        ]
        with self.assertRaises(InvalidInputError) as cm:
            parse_contours(raw)
        self.assertIn("segments", cm.exception.location)

    def test_unclosed_contour_rejected(self):
        raw = [
            [
                {"type": "line", "p0": {"x": 0, "y": 0}, "p1": {"x": 4, "y": 0}},
                {"type": "line", "p0": {"x": 4, "y": 0}, "p1": {"x": 4, "y": 4}},
                {"type": "line", "p0": {"x": 4, "y": 4}, "p1": {"x": 0, "y": 4}},
            ]
        ]
        with self.assertRaises(InvalidInputError) as cm:
            parse_contours(raw)
        self.assertIn("not closed", str(cm.exception))

    def test_disconnected_chain_rejected(self):
        raw = [
            [
                {"type": "line", "p0": {"x": 0, "y": 0}, "p1": {"x": 4, "y": 0}},
                {"type": "line", "p0": {"x": 4.0, "y": 1.0},
                 "p1": {"x": 4, "y": 4}},
                {"type": "line", "p0": {"x": 4, "y": 4}, "p1": {"x": 0, "y": 4}},
                {"type": "line", "p0": {"x": 0, "y": 4}, "p1": {"x": 0, "y": 0}},
            ]
        ]
        with self.assertRaises(InvalidInputError) as cm:
            parse_contours(raw)
        self.assertIn("does not meet", str(cm.exception))
        self.assertIn("segments[1]", cm.exception.location)

    def test_zero_length_line_rejected(self):
        raw = square_raw()
        raw[0][2] = {"type": "line", "p0": {"x": 4, "y": 4},
                     "p1": {"x": 4, "y": 4}}
        with self.assertRaises(InvalidInputError):
            parse_contours(raw)

    def test_quad_coincident_endpoints_rejected(self):
        raw = [
            [
                {"type": "quad", "p0": {"x": 0, "y": 0}, "c": {"x": 2, "y": 3},
                 "p1": {"x": 0, "y": 0}},
                {"type": "line", "p0": {"x": 0, "y": 0}, "p1": {"x": 4, "y": 0}},
                {"type": "line", "p0": {"x": 4, "y": 0}, "p1": {"x": 0, "y": 0}},
            ]
        ]
        with self.assertRaises(InvalidInputError):
            parse_contours(raw)

    def test_unknown_segment_type(self):
        raw = square_raw()
        raw[0][0] = {"type": "cubic", "p0": {"x": 0, "y": 0},
                     "p1": {"x": 1, "y": 0}}
        with self.assertRaises(InvalidInputError):
            parse_contours(raw)

    def test_non_list_contours(self):
        with self.assertRaises(InvalidInputError):
            parse_contours({"x": 1})
        with self.assertRaises(InvalidInputError):
            rasterize_raw("nope", 8, 8)

    def test_validation_runs_before_rasterization(self):
        # Bad second contour: there must be no result object at all,
        # never a partially computed canvas.
        raw = square_raw() + square_raw()
        raw[1][0]["p0"]["x"] = math.nan
        with self.assertRaises(InvalidInputError):
            rasterize_raw(raw, 8, 8)

    def test_rasterize_rejects_non_contour_objects(self):
        with self.assertRaises(InvalidInputError):
            rasterize([{"type": "line"}], 8, 8)

    def test_max_protocol_limits_work(self):
        # Interface boundary: MAX_CONTOURS contours on a MAX_CANVAS grid,
        # each contour a curved ring (quads split into 2 y-pieces).  This
        # must run as plain Python without special handling and stay
        # comfortably fast for the offline kernel use case.
        import time
        contours = []
        for k in range(MAX_CONTOURS):
            x0 = 1 + (k % 4) * 15
            y0 = 1 + (k // 4) * 12
            contours.append(
                build_contour(
                    [(x0, y0), (x0 + 10, y0), (x0 + 10, y0 + 10),
                     (x0, y0 + 10)],
                    quads=((0, x0 + 5, y0 - 1.5),
                           (2, x0 + 5, y0 + 11.5)),
                )
            )
        t0 = time.perf_counter()
        result = rasterize(contours, MAX_CANVAS, MAX_CANVAS)
        elapsed = time.perf_counter() - t0
        self.assertEqual(result.width, MAX_CANVAS)
        self.assertEqual(result.height, MAX_CANVAS)
        # at least one interior pixel of the first ring is solid
        self.assertEqual(result.levels[y0 + 5][x0 + 5], 16)
        self.assertLess(elapsed, 15.0)  # generous CI bound, not a burn-in


class QueryAndBuilderTests(unittest.TestCase):
    def setUp(self):
        self.result = rasterize_raw(square_raw(), 8, 8)

    def test_pixel_index_validation(self):
        with self.assertRaises(InvalidInputError):
            self.result.coverage_at(8, 0)
        with self.assertRaises(InvalidInputError):
            self.result.coverage_at(0, -1)
        with self.assertRaises(InvalidInputError):
            self.result.coverage_at(True, 0)

    def test_subsample_index_validation(self):
        with self.assertRaises(InvalidInputError):
            self.result.winding_at(3, 3, 4, 0)
        with self.assertRaises(InvalidInputError):
            self.result.winding_at(3, 3, 0, -1)
        with self.assertRaises(InvalidInputError):
            self.result.winding_at(3, 3, 0, True)

    def test_build_contour_basic_and_validation(self):
        con = build_contour([(0, 0), (4, 0), (4, 4), (0, 4)])
        result = rasterize([con], 6, 6)
        self.assertEqual(result.coverage_at(2, 2), 1.0)

        with self.assertRaises(InvalidInputError):
            build_contour([(0, 0), (4, 0)])
        with self.assertRaises(InvalidInputError):
            build_contour([(0, 0), (4, 0), (4, 4)],
                          quads=[(7, 1, 1)])
        with self.assertRaises(InvalidInputError):
            build_contour(
                [(0, 0), (4, 0), (4, 4)],
                quads=[(0, 1, 1), (0, 2, 2)],
            )
        with self.assertRaises(InvalidInputError):
            build_contour([(0, 0), (4, 0), (4, 4)], quads=[(True, 1, 1)])


if __name__ == "__main__":
    unittest.main()
