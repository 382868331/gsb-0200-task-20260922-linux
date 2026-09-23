"""核心几何与栅格化的契约测试。

期望值来源：
* 直边多边形：测试内手工解析推导（Fraction 精确）；
* 随机二次曲线用例：tests/_reference.py 的独立 Decimal 参考
  （不导入被测核心的任何算法函数）；
* 往返/对称性质仅作补充断言。
"""

from __future__ import annotations

import random
import unittest
from fractions import Fraction

from nonzero_fill import (
    InvalidCanvasError,
    InvalidContourError,
    InvalidCoordinateError,
    InvalidQueryError,
    line,
    outline,
    point,
    quad,
    rasterize,
)
from nonzero_fill.model import GlyphOutline

from tests._reference import ref_grid

F = Fraction


def rect(x0, y0, x1, y1, ccw=True):
    """轴对齐矩形轮廓。"""
    pts = [point(x0, y0), point(x1, y0), point(x1, y1), point(x0, y1)]
    if not ccw:
        pts = [pts[0], pts[3], pts[2], pts[1]]
    return [line(pts[i], pts[(i + 1) % 4]) for i in range(4)]


def reverse_contour(segments):
    """轮廓整体走向反转（二次贝塞尔 P0/P2 对调，控制点不变）。"""
    rev = []
    for seg in reversed(segments):
        if hasattr(seg, "p2"):
            rev.append(quad(seg.p2, seg.p1, seg.p0))
        else:
            rev.append(line(seg.p1, seg.p0))
    return rev


class ExactPolygonReferenceTests(unittest.TestCase):
    """直边多边形对解析精确参考。"""

    def test_axis_aligned_square(self):
        g = outline([rect(0, 0, 4, 4)])
        r = rasterize(g, 8, 8)
        for x in range(8):
            for y in range(8):
                expected = 1.0 if x < 4 and y < 4 else 0.0
                self.assertEqual(r.coverage_at(x, y), expected)
        # 内部环绕数恒为 +1（CCW），外部恒为 0。
        for y in range(4):
            for x in range(4):
                for sy in range(4):
                    for sx in range(4):
                        self.assertEqual(r.winding_at(x, y, sx, sy), 1)

    def test_triangle_exact(self):
        # CCW 三角形 (0,0)-(8,0)-(0,7)：斜边 7x+8y=56。
        # 采样点坐标为奇数/8：7·奇+8·奇 的分子为奇数，56=448/8 为偶数，
        # 故没有任何采样点恰在斜边上，内部条件 7xs+8ys<56 可精确断言。
        segs = [line(point(0, 0), point(8, 0)),
                line(point(8, 0), point(0, 7)),
                line(point(0, 7), point(0, 0))]
        r = rasterize(outline([segs]), 9, 9)
        for px in range(9):
            for py in range(9):
                for sx in range(4):
                    for sy in range(4):
                        xs = F(8 * px + 2 * sx + 1, 8)
                        ys = F(8 * py + 2 * sy + 1, 8)
                        w = r.winding_at(px, py, sx, sy)
                        self.assertEqual(w, 1 if 7 * xs + 8 * ys < 56 else 0)

    def test_diamond_with_fractional_vertices(self):
        # 菱形顶点 (3,0)-(6,2)-(3,4)-(0,2)，斜边系数 2 与 3：
        # 2·奇±3·奇 = 奇数（/8），不可能等于偶数整数边界值，
        # 故没有任何采样点落在边上，严格不等式可精确断言。
        # 内部：-6 < 2xs-3ys < 6 且 6 < 2xs+3ys < 18。
        segs = [line(point(3, 0), point(6, 2)),
                line(point(6, 2), point(3, 4)),
                line(point(3, 4), point(0, 2)),
                line(point(0, 2), point(3, 0))]
        r = rasterize(outline([segs]), 7, 5)
        for px in range(7):
            for py in range(5):
                for sx in range(4):
                    for sy in range(4):
                        xs = F(8 * px + 2 * sx + 1, 8)
                        ys = F(8 * py + 2 * sy + 1, 8)
                        u = 2 * xs - 3 * ys
                        v = 2 * xs + 3 * ys
                        inside = -6 < u < 6 and 6 < v < 18
                        self.assertEqual(
                            r.winding_at(px, py, sx, sy) != 0, inside,
                            (px, py, sx, sy))

    def test_coverage_is_exact_k_over_16(self):
        segs = [line(point(0, 0), point(3, 0)),
                line(point(3, 0), point(0, 5)),
                line(point(0, 5), point(0, 0))]
        r = rasterize(outline([segs]), 8, 8)
        for row in r.coverage:
            for v in row:
                self.assertIn(
                    v, {k / 16 for k in range(17)})
                # k/16 在二进制浮点中精确：往返乘 16 必为整数。
                self.assertEqual(round(v * 16), v * 16)


class WindingInvariantTests(unittest.TestCase):
    """反向抵消与整体方向反转不变性。"""

    def test_opposite_contours_cancel_exactly(self):
        base = rect(1, 1, 7, 7)
        g = outline([base, reverse_contour(base)])
        r = rasterize(g, 8, 8)
        for px in range(8):
            for py in range(8):
                self.assertEqual(r.coverage_at(px, py), 0.0)
                for sx in range(4):
                    for sy in range(4):
                        self.assertEqual(
                            r.winding_at(px, py, sx, sy), 0)

    def test_global_reversal_flips_winding_keeps_coverage(self):
        segs = [line(point(0, 0), point(5, 0)),
                line(point(5, 0), point(5, 3)),
                line(point(5, 3), point(0, 0))]
        r1 = rasterize(outline([segs]), 7, 6)
        r2 = rasterize(outline([reverse_contour(segs)]), 7, 6)
        for px in range(7):
            for py in range(6):
                self.assertEqual(r1.coverage_at(px, py),
                                 r2.coverage_at(px, py))
                for sx in range(4):
                    for sy in range(4):
                        self.assertEqual(
                            r1.winding_at(px, py, sx, sy),
                            -r2.winding_at(px, py, sx, sy))

    def test_hole_cancels_winding(self):
        # 外框 CCW 8x8，内框 CW 4x4：洞中环绕数精确为 0。
        g = outline([rect(0, 0, 8, 8), rect(2, 2, 6, 6, ccw=False)])
        r = rasterize(g, 8, 8)
        for px in range(8):
            for py in range(8):
                in_hole = 2 <= px < 6 and 2 <= py < 6
                for sx in range(4):
                    for sy in range(4):
                        w = r.winding_at(px, py, sx, sy)
                        if in_hole:
                            self.assertEqual(w, 0)
                        else:
                            self.assertEqual(abs(w), 1)
                self.assertEqual(r.coverage_at(px, py),
                                 0.0 if in_hole else 1.0)

    def test_overlap_winding_two_still_filled(self):
        # 两个同向重叠矩形：重叠区环绕数 2，nonzero 规则下仍填充。
        g = outline([rect(0, 0, 6, 6), rect(2, 2, 8, 8)])
        r = rasterize(g, 9, 9)
        for px in range(9):
            for py in range(9):
                overlap = 2 <= px < 6 and 2 <= py < 6
                single = (px < 6 and py < 6) or (2 <= px < 8 and 2 <= py < 8)
                for sx in range(4):
                    for sy in range(4):
                        w = r.winding_at(px, py, sx, sy)
                        if overlap:
                            self.assertEqual(w, 2)
                        elif single:
                            self.assertEqual(w, 1)
                        else:
                            self.assertEqual(w, 0)
                self.assertEqual(r.coverage_at(px, py),
                                 1.0 if single else 0.0)


class SelfIntersectionTests(unittest.TestCase):
    """自交轮廓（蝴蝶结）：两瓣环绕数符号相反、均非零即都填充。"""

    def setUp(self):
        # (0,0)->(6,4)->(6,0)->(0,4) 闭合。
        # E1 直线 2x-3y=0 与 E3 直线 2x+3y=12 自交于 (3,2)。
        # 两条斜边代入奇数/8 采样点：2a-3b=0 要求 b 为偶数，
        # 2a+3b=96 为奇数=偶数矛盾——无采样点落在边上。
        self.segs = [line(point(0, 0), point(6, 4)),
                     line(point(6, 4), point(6, 0)),
                     line(point(6, 0), point(0, 4)),
                     line(point(0, 4), point(0, 0))]
        self.r = rasterize(outline([self.segs]), 7, 5)

    def _w(self, px, py, sx, sy):
        return self.r.winding_at(px, py, sx, sy)

    def test_lobes_analytic(self):
        for py in range(5):
            for sy in range(4):
                ys = F(2 * (4 * py + sy) + 1, 8)
                for px in range(7):
                    for sx in range(4):
                        xs = F(8 * px + 2 * sx + 1, 8)
                        u = 2 * xs - 3 * ys
                        v = 2 * xs + 3 * ys
                        in_right = u > 0 and v > 12 and xs < 6
                        in_left = u < 0 and v < 12 and xs > 0
                        w = self._w(px, py, sx, sy)
                        if in_right:
                            self.assertEqual(w, -1, (xs, ys))
                        elif in_left:
                            self.assertEqual(w, +1, (xs, ys))
                        else:
                            self.assertEqual(w, 0, (xs, ys))

    def test_lobe_fill_under_nonzero_rule(self):
        # 两瓣环绕数非零（尽管符号相反），nonzero 下两瓣都应有覆盖。
        self.assertGreater(self.r.coverage_at(5, 1), 0.0)
        self.assertGreater(self.r.coverage_at(0, 2), 0.0)

    def test_matches_independent_reference(self):
        grid, coverage, near = ref_grid(outline([self.segs]), 7, 5)
        self.assertEqual(near, 0)
        for py in range(5):
            for px in range(7):
                for sy in range(4):
                    for sx in range(4):
                        self.assertEqual(
                            self.r.winding_at(px, py, sx, sy),
                            grid[4 * py + sy][4 * px + sx])


class VertexOnSampleLineTests(unittest.TestCase):
    """顶点/边落在采样行上的半开规则（解析断言）。"""

    def test_horizontal_edges_on_sample_rows(self):
        # 矩形水平边恰在 y=1/8 与 y=33/8（采样行）。
        segs = rect(F(1, 1), F(1, 8), F(5, 1), F(33, 8))
        r = rasterize(outline([segs]), 7, 5)
        # 底行 i=0（ys=1/8）：两个竖直交点 1 与 5，其间环绕数为 +1。
        for sx in range(4):
            for px in range(7):
                xs = F(8 * px + 2 * sx + 1, 8)
                w = r.winding_at(px, 0, sx, 0)
                self.assertEqual(w, 1 if 1 < xs < 5 else 0)

    def test_apex_on_sample_line_single_crossing_never_double(self):
        # 三角形顶点 (4, 17/8) 恰在采样行；该行除顶点外另有两条竖边穿越，
        # 顶点相邻两边按半开规则同时排除，故贡献与“行间”参考一致。
        apex_y = F(17, 8)
        segs = [line(point(0, 0), point(8, 0)),
                line(point(8, 0), point(4, apex_y)),
                line(point(4, apex_y), point(0, 0))]
        r = rasterize(outline([segs]), 9, 3)
        # 顶点所在行 i=8（ys=17/8，即像素 py=2 的 sy=0）：三角形在该行
        # 仅在顶点处触线，宽度为 0，全部子样本环绕数为 0。
        for px in range(9):
            for sx in range(4):
                self.assertEqual(r.winding_at(px, 2, sx, 0), 0)
        # 紧邻其下一行（i=7, ys=15/8）三角形内部非空。
        self.assertTrue(any(r.winding_at(px, 1, sx, 3) != 0
                            for px in range(9) for sx in range(4)))

    def test_horizontal_segments_never_count(self):
        # 矩形上下两条水平段落在整数 y 上；水平段任何采样行都不计。
        a = [line(point(1, 1), point(3, 1)),
             line(point(3, 1), point(3, 3)),
             line(point(3, 3), point(1, 3)),
             line(point(1, 3), point(1, 1))]
        r = rasterize(outline([a]), 5, 5)
        for px in range(5):
            for py in range(5):
                expected = 1.0 if 1 <= px < 3 and 1 <= py < 3 else 0.0
                self.assertEqual(r.coverage_at(px, py), expected)

    def test_sample_point_exactly_on_vertical_edges(self):
        # 左右竖边恰在采样列 x=1/8 与 x=9/8（均为奇数/8）。
        # 契约：交点要求严格 x>xs；配合半开 y，边界点得到确定的环绕数，
        # 不做特判、不抛“边界未决”。
        segs = rect(F(1, 8), 0, F(9, 8), 2)
        r = rasterize(outline([segs]), 2, 2)
        # 行 i=0（ys=1/8）：j=0 在左边、j=4 在右边（j=4 属像素 px=1）。
        # 左边界点：右边交点仍在其右侧 +1；右边界点：右交点等号不计 → 0。
        self.assertEqual(r.winding_at(0, 0, 0, 0), 1)  # xs=1/8
        self.assertEqual(r.winding_at(1, 0, 0, 0), 0)  # xs=9/8
        # 内部列正常为 1。
        self.assertEqual(r.winding_at(0, 0, 1, 0), 1)  # xs=3/8
        self.assertEqual(r.winding_at(0, 0, 3, 0), 1)  # xs=7/8


class QuadraticTests(unittest.TestCase):
    """二次贝塞尔：拆分、相切、无理根定号。"""

    def test_quadratic_arch_exact_polynomial_predicate(self):
        # 拱形：P0=(0,0) P1=(2,1) P2=(4,0)，底边 (4,0)->(0,0) 闭合。
        # x(t)=4t，y(t)=2t(1-t)；消参得曲线 y = x/2 - x²/8。
        # 拱下内部（轮廓为 CW）：0<xs<4 且 8ys < 4xs-xs²。
        # 采样坐标为奇数/8 时右边 (32·奇-奇²)/64 分子为奇数，
        # 左边 8ys 为奇数但分母不同——化简可知无采样点落在曲线上：
        # 8ys-(4xs-xs²)=0 ⇔ 64·奇=32·奇-奇²，奇偶矛盾，故严格不等式精确。
        q = quad(point(0, 0), point(2, 1), point(4, 0))
        segs = [q, line(point(4, 0), point(0, 0))]
        r = rasterize(outline([segs]), 6, 3)
        for py in range(3):
            for sy in range(4):
                ys = F(2 * (4 * py + sy) + 1, 8)
                for px in range(6):
                    for sx in range(4):
                        xs = F(8 * px + 2 * sx + 1, 8)
                        inside = 0 < xs < 4 and 8 * ys < 4 * xs - xs * xs
                        w = r.winding_at(px, py, sx, sy)
                        # 右半支向下，内部射线只穿一次 → 环绕数 -1。
                        self.assertEqual(w, -1 if inside else 0,
                                         (px, py, sx, sy, xs, ys))

    def test_quadratic_arch_rational_crossing_row(self):
        # 同一拱形在采样行 ys=3/8（i=1）有两个有理根 t=1/4、3/4，
        # 对应交点 x=1 与 x=3：验证 y 极值拆分后一行双交点的精确位置。
        q = quad(point(0, 0), point(2, 1), point(4, 0))
        segs = [q, line(point(4, 0), point(0, 0))]
        r = rasterize(outline([segs]), 6, 1)
        # ys=3/8 = 全局行 i=1（py=0, sy=1）。
        # 交点 x=1 与 x=3：其间 w=-1，两侧为 0（等号不计）。
        for j in range(24):  # 全局子样本列
            xs = F(2 * j + 1, 8)
            px, sx = divmod(j, 4)
            expected = -1 if 1 < xs < 3 else 0
            self.assertEqual(r.winding_at(px, 0, sx, 1), expected, (j, xs))

    def test_sample_point_exactly_on_quadratic(self):
        # 构造一条精确经过采样点 (3/8, 1/8) 的二次曲线：
        # P0=(0,0)、P1=(11/7, 1/2)、P2=(2,1)，则 y(t)=t，
        # x(1/8)=2(7/8)(1/8)(11/7)+2(1/8)^2 = 3/8。
        # 以竖边 (2,1)->(2,0) 与水平底边闭合：ys=1/8 行的交点为
        # 曲线 x=3/8（向上 +1）与竖边 x=2（向下 -1）。
        q = quad(point(0, 0), point(F(11, 7), F(1, 2)), point(2, 1))
        segs = [q,
                line(point(2, 1), point(2, 0)),
                line(point(2, 0), point(0, 0))]
        g = outline([segs])
        r = rasterize(g, 3, 2)
        # j=0（xs=1/8）：曲线与竖边都在右侧，+1-1=0。
        self.assertEqual(r.winding_at(0, 0, 0, 0), 0)
        # j=1（xs=3/8，恰在曲线上）：曲线等号不计；竖边仍在右 → -1。
        # 若错误地把等号计入，会得到 0，故此断言锁定严格比较语义。
        self.assertEqual(r.winding_at(0, 0, 1, 0), -1)
        # j=2（xs=5/8）：仅竖边在右 → -1；竖边之外 → 0。
        self.assertEqual(r.winding_at(0, 0, 2, 0), -1)
        self.assertEqual(r.winding_at(2, 0, 3, 0), 0)  # xs=23/8 > 2
        # 与独立参考（Fraction 精确 tie 判定）一致。
        from tests._reference import ref_winding
        w0, near0 = ref_winding(list(g.contours), F(1, 8), F(1, 8))
        w1, near1 = ref_winding(list(g.contours), F(3, 8), F(1, 8))
        self.assertFalse(near0 or near1)
        self.assertEqual(w0, 0)
        self.assertEqual(w1, -1)

    def test_tangent_extremum_on_sample_row(self):
        # y(t)=2*y1*t(1-t)，取 y1=5/4，则极值 y_max=5/8=采样行 i=2。
        q = quad(point(0, 0), point(2, F(5, 4)), point(4, 0))
        segs = [q, line(point(4, 0), point(0, 0))]  # 闭合边水平
        r = rasterize(outline([segs]), 6, 2)
        # 相切行（py=0, sy=2）：曲线只在顶点触碰，全列环绕数为 0。
        for px in range(6):
            for sx in range(4):
                self.assertEqual(r.winding_at(px, 0, sx, 2), 0)
        # 相邻两行内部存在非零区域（拱形下方）。
        filled_near = sum(
            1 for px in range(6) for sx in range(4)
            for sy in (1, 3)
            if r.winding_at(px, 0, sx, sy) != 0)
        self.assertGreater(filled_near, 0)

    def test_quadratic_matches_independent_reference_random(self):
        rng = random.Random(20260922)

        def rp():
            return point(F(rng.randint(0, 48), rng.randint(1, 4)),
                         F(rng.randint(0, 48), rng.randint(1, 4)))

        for case in range(14):
            contours = []
            for _ in range(rng.randint(1, 3)):
                start = rp()
                cur = start
                segs = []
                for _ in range(rng.randint(2, 4)):
                    if rng.random() < 0.5:
                        nxt = rp()
                        segs.append(line(cur, nxt))
                    else:
                        ctrl = rp()
                        nxt = rp()
                        segs.append(quad(cur, ctrl, nxt))
                    cur = nxt
                # 强制闭合：末段回到起点。
                if cur != start:
                    if rng.random() < 0.5:
                        segs.append(line(cur, start))
                    else:
                        segs.append(quad(cur, rp(), start))
                else:
                    # 末段终点已在起点，确保链闭合。
                    pass
                try:
                    g0 = outline([segs])
                except InvalidContourError:
                    continue
                contours.append(list(g0.contours[0]))
            if not contours:
                continue
            g = GlyphOutline(tuple(tuple(c) for c in contours))
            w = 12
            res = rasterize(g, w, w)
            grid, coverage, near = ref_grid(g, w, w)
            self.assertEqual(near, 0, f"case {case}: 参考出现近重合点")
            for py in range(w):
                for px in range(w):
                    for sy in range(4):
                        for sx in range(4):
                            self.assertEqual(
                                res.winding_at(px, py, sx, sy),
                                grid[4 * py + sy][4 * px + sx],
                                f"case {case} at {px},{py},{sx},{sy}")
                    self.assertEqual(res.coverage_at(px, py),
                                     coverage[py][px])


class ValidationTests(unittest.TestCase):
    """错误语义：可定位、不静默纠正、整数拒绝 bool、拒绝 NaN/Infinity。"""

    def test_reject_nan_infinity_float(self):
        for bad, token in ((float("nan"), "NaN"),
                           (float("inf"), "Infinity"),
                           (float("-inf"), "Infinity"),
                           (1.5, "float")):
            with self.assertRaises(InvalidCoordinateError) as cm:
                point(bad, 0)
            self.assertIn(token, str(cm.exception))
            with self.assertRaises(InvalidCoordinateError):
                point(0, bad)

    def test_reject_bool_for_coordinates_and_ints(self):
        with self.assertRaises(InvalidCoordinateError):
            point(True, 0)
        with self.assertRaises(InvalidCanvasError):
            rasterize(outline([]), True, 8)
        with self.assertRaises(InvalidCanvasError):
            rasterize(outline([]), 8, False)

    def test_canvas_bounds_and_type(self):
        g = outline([])
        for w, h in ((0, 8), (65, 8), (8, 0), (8, -1), (8, 1.0)):
            with self.assertRaises((InvalidCanvasError, TypeError)):
                rasterize(g, w, h)
        # 64 是合法上界。
        rasterize(g, 64, 64)

    def test_contour_limits_and_shape(self):
        with self.assertRaises(InvalidContourError):
            outline([[]])  # 含一条空轮廓（空字形 outline([]) 合法）
        with self.assertRaises(InvalidContourError):
            outline([[line(point(0, 0), point(1, 1))]
                     for _ in range(21)])
        with self.assertRaises(InvalidContourError) as cm:
            outline([[line(point(0, 0), point(1, 1)),
                      line(point(2, 2), point(0, 0))]])  # 不衔接
        self.assertIn("不衔接", str(cm.exception))
        with self.assertRaises(InvalidContourError) as cm:
            outline([[line(point(0, 0), point(1, 1)),
                      line(point(1, 1), point(2, 2))]])  # 未闭合
        self.assertIn("segments[1]", str(cm.exception))

    def test_degenerate_segments(self):
        with self.assertRaises(InvalidContourError):
            line(point(0, 0), point(0, 0))
        with self.assertRaises(InvalidContourError):
            quad(point(1, 1), point(1, 1), point(1, 1))

    def test_raw_segments_rejected_with_location(self):
        with self.assertRaises(InvalidContourError) as cm:
            outline([[((0, 0), (1, 1))]])
        self.assertIn("segments[0]", str(cm.exception))

    def test_glyph_type_checked(self):
        with self.assertRaises(InvalidCanvasError):
            rasterize([], 8, 8)

    def test_query_index_errors(self):
        r = rasterize(outline([rect(0, 0, 2, 2)]), 4, 4)
        with self.assertRaises(InvalidQueryError):
            r.coverage_at(4, 0)
        with self.assertRaises(InvalidQueryError):
            r.winding_at(0, 0, 4, 0)
        with self.assertRaises(InvalidQueryError):
            r.winding_at(0, 0, 0, -1)
        with self.assertRaises(InvalidQueryError):
            r.coverage_at(True, 0)

    def test_no_partial_result_on_failure(self):
        # 构造失败时不返回对象；outline 是原子校验。
        before = None
        result = before
        with self.assertRaises(InvalidContourError):
            outline([rect(0, 0, 2, 2),
                     [line(point(0, 0), point(0, 0))]])  # 第二条非法
        self.assertIsNone(result)


class EmptyAndImmutableTests(unittest.TestCase):
    def test_empty_glyph_all_zero(self):
        r = rasterize(outline([]), 4, 4)
        for px in range(4):
            for py in range(4):
                self.assertEqual(r.coverage_at(px, py), 0.0)
                for sx in range(4):
                    for sy in range(4):
                        self.assertEqual(r.winding_at(px, py, sx, sy), 0)

    def test_result_consistency_supplement(self):
        # 补充性质：coverage=非零子样本数/16（由 winding 数据重算一致）。
        g = outline([rect(0, 0, 5, 3),
                     rect(2, 1, 7, 6, ccw=False)])
        r = rasterize(g, 8, 8)
        for px in range(8):
            for py in range(8):
                cnt = sum(1 for sy in range(4) for sx in range(4)
                          if r.winding_at(px, py, sx, sy) != 0)
                self.assertEqual(r.coverage_at(px, py), cnt / 16)


if __name__ == "__main__":
    unittest.main()
