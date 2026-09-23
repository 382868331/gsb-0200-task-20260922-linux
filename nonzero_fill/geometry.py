"""精确有理几何核心：段表示、二次曲线 y 单调拆分、半开规则交点判定。

坐标全部为 :class:`fractions.Fraction`，几何定号全程精确（无浮点、无迭代）。

半开 y 规则（直线段与单调曲线段共用，标准 winding-number 约定）：

* 方向向上（走向 y 递增，起点低、终点高）：``y_start <= y0 < y_end`` 时 +1；
* 方向向下（起点高、终点低）：``y_end <= y0 < y_start`` 时 -1；
* 即“向上边包含起点、向下边包含终点”；水平段永远不计。

于是顶点恰在采样线上时相邻两边合计只产生一次穿越；在局部最低/最高
顶点（相切触碰）处两边贡献 +1 与 -1 合计为 0，与把采样行微扰到
顶点上/下方的极限一致，无需特判。

二次贝塞尔 ``B(t)=(1-t)^2 P0+2(1-t)t P1+t^2 P2``，
``y(t)=c t^2+b t+y0``，``c=y0-2y1+y2``，``b=2(y1-y0)``，
内部极值参数 ``t*=-b/(2c)``，用 de Casteljau 在 t* 处精确拆分。
单调段上 ``y(t)=y0`` 至多一个根；根可能是二次无理数，其 x 坐标与
采样列的大小比较由栅格器在二次扩域 ``Q(sqrt(D))`` 内精确定号
（见 :mod:`nonzero_fill.rasterizer`），本模块只负责拆分与区间判定。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class LineSegment:
    """直线段 P0 -> P1。"""

    p0: Tuple[Fraction, Fraction]
    p1: Tuple[Fraction, Fraction]


@dataclass(frozen=True)
class QuadraticSegment:
    """二次贝塞尔段 P0 -> P2，控制点 P1。"""

    p0: Tuple[Fraction, Fraction]
    p1: Tuple[Fraction, Fraction]
    p2: Tuple[Fraction, Fraction]


Segment = LineSegment | QuadraticSegment


@dataclass(frozen=True)
class MonoPiece:
    """二次曲线拆出的 y 单调原子段（局部参数 u∈[0,1]）。

    坐标多项式 ``x(u)=a2 u^2+a1 u+x0``、``y(u)=c2 u^2+c1 u+y0``。
    ``upward`` 为走向（True 表示随 u 的 y 严格递增）。
    """

    a2: Fraction
    a1: Fraction
    x0: Fraction
    c2: Fraction
    c1: Fraction
    y0: Fraction
    upward: bool

    def y_end(self) -> Fraction:
        return self.c2 + self.c1 + self.y0

    def x_end(self) -> Fraction:
        return self.a2 + self.a1 + self.x0


def split_quadratic(seg: QuadraticSegment) -> List[MonoPiece]:
    """把二次贝塞尔在内部 y 极值处拆成 y 单调原子段（精确）。"""
    (x0, y0), (x1, y1), (x2, y2) = seg.p0, seg.p1, seg.p2
    c = y0 - 2 * y1 + y2
    b = 2 * (y1 - y0)

    def piece(qx0: Fraction, qy0: Fraction,
              qx1: Fraction, qy1: Fraction,
              qx2: Fraction, qy2: Fraction) -> MonoPiece:
        return MonoPiece(
            qx0 - 2 * qx1 + qx2,
            2 * (qx1 - qx0),
            qx0,
            qy0 - 2 * qy1 + qy2,
            2 * (qy1 - qy0),
            qy0,
            qy2 > qy0,
        )

    if c == 0:
        # y 线性（含水平退化，水平段在求交时自然被区间规则排除）。
        return [piece(x0, y0, x1, y1, x2, y2)]

    t_star = -b / (2 * c)
    if t_star <= 0 or t_star >= 1:
        # 极值在参数区间外（或恰在端点），开区间内 y 单调。
        return [piece(x0, y0, x1, y1, x2, y2)]

    # de Casteljau 精确拆分。
    om = 1 - t_star
    xa, ya = om * x0 + t_star * x1, om * y0 + t_star * y1
    xb, yb = om * x1 + t_star * x2, om * y1 + t_star * y2
    xm, ym = om * xa + t_star * xb, om * ya + t_star * yb
    return [
        piece(x0, y0, xa, ya, xm, ym),
        piece(xm, ym, xb, yb, x2, y2),
    ]


def line_hits_row(seg: LineSegment, y0: Fraction) -> bool:
    """直线段是否按半开 y 规则命中采样行 y0（水平段恒 False）。"""
    sy0, sy1 = seg.p0[1], seg.p1[1]
    if sy1 > sy0:  # 向上：含起点（低端）
        return sy0 <= y0 < sy1
    if sy1 < sy0:  # 向下：含终点（低端）
        return sy1 <= y0 < sy0
    return False


def piece_hits_row(p: MonoPiece, y0: Fraction) -> bool:
    """单调曲线段是否按半开 y 规则命中采样行 y0。"""
    yl, yh = p.y0, p.y_end()
    if p.upward:  # 起点为低端：含起点
        return yl <= y0 < yh
    return yh <= y0 < yl  # 终点为低端：含终点


def segments_of(contours: Sequence[Sequence[Segment]]
                ) -> Tuple[List[LineSegment], List[MonoPiece]]:
    """展开全部轮廓：直线段原物返回，二次段预拆为单调原子段。"""
    lines: List[LineSegment] = []
    pieces: List[MonoPiece] = []
    for contour in contours:
        for seg in contour:
            if isinstance(seg, LineSegment):
                lines.append(seg)
            else:
                pieces.extend(split_quadratic(seg))
    return lines, pieces
