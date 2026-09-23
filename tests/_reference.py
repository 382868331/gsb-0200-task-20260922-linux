"""测试专用独立参考实现（不调用被测核心的任何算法）。

与 ``nonzero_fill`` 的关系：仅共享输入数据结构（GlyphOutline 内的
LineSegment/QuadraticSegment 数据类，只读取 Fraction 坐标），环绕数
完全由本文件独立计算：

* 直线段：Fraction 精确求交；
* 二次贝塞尔：独立按 de Casteljau 在 y 极值拆分，根用 Decimal(60 位)
  的二次公式独立求解；y 的半开判定仍用 Fraction 精确比较。

对采样点恰在曲线上（x 差近似 0）的情形返回 ``AMBIGUOUS``，由调用测试
按本题契约（射线交点要求严格 x>xs，等号不计入）另行精确构造断言。
"""

from __future__ import annotations

from decimal import Decimal, getcontext
from fractions import Fraction
from typing import List, Optional, Tuple

from nonzero_fill.geometry import LineSegment, QuadraticSegment

getcontext().prec = 60

_D0 = Decimal(0)
_D1 = Decimal(1)
_NEAR = Decimal("1e-42")


def _dec(f: Fraction) -> Decimal:
    return Decimal(f.numerator) / Decimal(f.denominator)


def _half_open(y_lo: Fraction, y_hi: Fraction, ys: Fraction) -> bool:
    # 半开约定与走向无关：几何 y 范围一律含低端、不含高端；
    # 方向（+1/-1）由调用方按边的走向另行决定。
    return y_lo <= ys < y_hi


def _casteljau(p0, p1, p2, t: Fraction):
    om = 1 - t
    a = (om * p0[0] + t * p1[0], om * p0[1] + t * p1[1])
    b = (om * p1[0] + t * p2[0], om * p1[1] + t * p2[1])
    m = (om * a[0] + t * b[0], om * a[1] + t * b[1])
    return (p0, a, m), (m, b, p2)


def _quad_pieces(seg: QuadraticSegment):
    """独立重写的 y 极值拆分，返回 ((q0,q1,q2,upward), ...)。"""
    p0, p1, p2 = seg.p0, seg.p1, seg.p2
    c = p0[1] - 2 * p1[1] + p2[1]
    b = 2 * (p1[1] - p0[1])
    if c == 0:
        return [(p0, p1, p2, p2[1] > p0[1])]
    t_star = -b / (2 * c)
    if t_star <= 0 or t_star >= 1:
        return [(p0, p1, p2, p2[1] > p0[1])]
    left, right = _casteljau(p0, p1, p2, t_star)
    return [
        (left[0], left[1], left[2], left[2][1] > left[0][1]),
        (right[0], right[1], right[2], right[2][1] > right[0][1]),
    ]


def _cmp_x_line(seg: LineSegment, ys: Fraction, xs_frac: Fraction
                ) -> Optional[int]:
    """直线命中行时，x_hit 相对 xs 的精确符号（+1 右、0 等、-1 左）。"""
    (x0, y0), (x1, y1) = seg.p0, seg.p1
    t = (ys - y0) / (y1 - y0)
    x_hit = x0 + t * (x1 - x0)
    # x_hit 与采样列都是 Fraction，精确比较。
    return (x_hit > xs_frac) - (x_hit < xs_frac)


def _quadratic_exact_tie(q0, q1, q2, ys_frac: Fraction,
                         xs_frac: Fraction) -> bool:
    """独立的精确“曲线恰过采样点”判定（Fraction，不经过被测核心）。

    y(t)=p t^2+q t+y0，命中根 t0 满足 p t0^2+q t0+e=0（e=y0-ys）。
    消去二次项：x(t0)-xs = α t0 + β，其中
    α=a1-a2 q/p，β=x0-xs-a2 e/p；α=0 时要求 β=0，
    否则候选 t*=-β/α 必须落在 [0,1] 且精确满足 y(t*)=ys。
    """
    x0, y0 = q0
    x1, y1 = q1
    x2, y2 = q2
    p = y0 - 2 * y1 + y2
    q = 2 * (y1 - y0)
    a2 = x0 - 2 * x1 + x2
    a1 = 2 * (x1 - x0)
    e = y0 - ys_frac
    if p == 0:
        t0 = (ys_frac - y0) / q
        return a2 * t0 * t0 + a1 * t0 + x0 == xs_frac
    alpha = a1 - a2 * q / p
    beta = (x0 - xs_frac) - a2 * e / p
    if alpha == 0:
        return beta == 0
    t_star = -beta / alpha
    if not (0 <= t_star <= 1):
        return False
    return p * t_star * t_star + q * t_star + e == 0


def _cmp_x_quad_piece(q0, q1, q2, ys_frac: Fraction,
                      xs_dec: Decimal, xs_frac: Fraction
                      ) -> Tuple[Optional[int], Decimal]:
    """Decimal 独立求单调段根并比较 x；返回 (符号或 None, 最小间距)。"""
    x0d, y0d = _dec(q0[0]), _dec(q0[1])
    x1d, y1d = _dec(q1[0]), _dec(q1[1])
    x2d, y2d = _dec(q2[0]), _dec(q2[1])
    ysd = _dec(ys_frac)
    c = y0d - 2 * y1d + y2d
    b = 2 * (y1d - y0d)
    e = y0d - ysd
    if c == 0:
        # y 线性：t=(ys-y0)/b。
        if b == 0:
            return None, Decimal("Infinity")
        t_hit = (ysd - y0d) / b
        if not (_D0 - Decimal("1e-50") <= t_hit <= _D1 + Decimal("1e-50")):
            return None, Decimal("Infinity")
        t_hit = min(max(t_hit, _D0), _D1)
        om = _D1 - t_hit
        x_hit = om * om * x0d + 2 * om * t_hit * x1d + t_hit * t_hit * x2d
        delta = x_hit - xs_dec
        return (0 if delta == 0 else (1 if delta > 0 else -1)), abs(delta)
    disc = b * b - 4 * c * e
    if disc < 0:
        return None, Decimal("Infinity")
    root = disc.sqrt()
    candidates = [(-b + root) / (2 * c), (-b - root) / (2 * c)]
    t_hit = None
    for t in candidates:
        if -Decimal("1e-50") <= t <= 1 + Decimal("1e-50"):
            t_hit = min(max(t, _D0), _D1)
            break
    if t_hit is None:
        return None, Decimal("Infinity")
    om = _D1 - t_hit
    x_hit = om * om * x0d + 2 * om * t_hit * x1d + t_hit * t_hit * x2d
    delta = x_hit - xs_dec
    if abs(delta) < _NEAR:
        return 0, abs(delta)
    return (1 if delta > 0 else -1), abs(delta)


def ref_winding(contours, xs: Fraction, ys: Fraction,
                ) -> Tuple[int, bool]:
    """返回 (环绕数, 是否含与采样点近重合的交点)。

    near=True 表示 Decimal 无法以 1e-42 间距区分（可能恰在边界上），
    随机用例中不应出现；精确边界用例由测试直接按契约断言。
    """
    xs_dec = _dec(xs)
    w = 0
    near = False
    for contour in contours:
        for seg in contour:
            if isinstance(seg, LineSegment):
                y0, y1 = seg.p0[1], seg.p1[1]
                if y0 == y1:
                    continue
                upward = y1 > y0
                if not _half_open(min(y0, y1), max(y0, y1), ys):
                    continue
                cmp_x = _cmp_x_line(seg, ys, xs)
                if cmp_x and cmp_x > 0:
                    w += 1 if upward else -1
            else:
                for q0, q1, q2, upward in _quad_pieces(seg):
                    y_lo = min(q0[1], q2[1])
                    y_hi = max(q0[1], q2[1])
                    if y_lo == y_hi or not _half_open(y_lo, y_hi, ys):
                        continue
                    # 精确等号优先：曲线恰过采样点时按契约不计入，
                    # 也不算“近重合未决”（这是已证明的相等，不是算不动）。
                    if _quadratic_exact_tie(q0, q1, q2, ys, xs):
                        continue
                    cmp_x, margin = _cmp_x_quad_piece(
                        q0, q1, q2, ys, xs_dec, xs)
                    if cmp_x is None:
                        continue
                    if margin < _NEAR:
                        near = True
                    if cmp_x > 0:
                        w += 1 if upward else -1
    return w, near


def ref_grid(glyph, width: int, height: int):
    """独立计算全 4x4 子采样网格的环绕数与覆盖（返回近重合计数）。"""
    grid: List[List[int]] = [
        [0] * (4 * width) for _ in range(4 * height)]
    near_count = 0
    contours = list(glyph.contours)
    for i in range(4 * height):
        ys = Fraction(2 * i + 1, 8)
        for j in range(4 * width):
            xs = Fraction(2 * j + 1, 8)
            w, near = ref_winding(contours, xs, ys)
            grid[i][j] = w
            near_count += int(near)
    coverage = tuple(
        tuple(
            sum(1 for sy in range(4) for sx in range(4)
                if grid[4 * py + sy][4 * px + sx] != 0) / 16
            for px in range(width))
        for py in range(height))
    return grid, coverage, near_count
