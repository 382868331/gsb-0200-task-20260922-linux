"""4x4 子采样非零环绕栅格器（全程精确定号，无浮点参与判定）。

采样点：像素 (px, py) 内第 (sx, sy) 个子样本位于
``xs=(8px+2sx+1)/8``、``ys=(8py+2sy+1)/8``（sx,sy ∈ {0,1,2,3}）；
等价地，全局子样本列/行 j=4px+sx、i=4py+sy 处坐标为 (2j+1)/8、(2i+1)/8。

环绕数约定：自采样点向 +x 方向发水平射线；向上（轮廓走向 y 递增）
的交点贡献 +1，向下贡献 -1，交点 x 严格大于 xs 才计入，故
``winding = Σ dir · [x_hit > xs]``。覆盖 = 16 个子样本中 winding≠0
的数量 / 16（k/16 在二进制浮点中精确可表示，输出零舍入误差）。

交点 x 可能是二次无理根。直线段的根恒为 Fraction；二次单调段的根
在判别式 D 为完全平方时也是 Fraction，否则落在二次扩域 Q(√D)，
本模块不消根、不近似：所有“x_hit 与 xs 的大小比较”通过
``sign(A + k√D)`` 的精确代数定号完成（至多一次整数平方比较，
D 与系数全为 Fraction）。因此不存在“搜索预算耗尽”——有无交点由
半开区间严格判定，能定号则定号，不存在迭代、容差与无解歧义。
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import List, Tuple

from .errors import InvalidCanvasError, InvalidQueryError
from .geometry import (
    LineSegment,
    MonoPiece,
    line_hits_row,
    piece_hits_row,
    segments_of,
)
from .model import GlyphOutline

SUBSAMPLES = 4
MAX_CANVAS = 64

# 行交点的内部表示：
#   (0, dir, x)              有理交点 x: Fraction
#   (1, dir, p, R0, M, D)    无理交点，见下
# 对单调段 y(u)=p u^2+q u+y0、x(u)=a2 u^2+a1 u+x0、根
# t_s=(-q+s√D)/(2p)，D=q^2-4p(y0-ys)，e=y0-ys，C=a1 p-a2 q。
# 直接代入 t_s（r=-q+s√D）：
#   2p^2 (x_hit-xs) = a2 r^2/2 + p a1 r + 2p^2(x0-xs)
#                   = (R0-2p^2 xs) + M√D，
#   R0 = 2p^2 x0 - p a1 q + a2 q^2 - 2p a2 e，M = s C。
# 2p^2 恒正，故 sign(x_hit-xs)=sign((R0-2p^2 xs)+M√D)，无需 sign(p) 因子。


def _is_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise InvalidCanvasError(f"{name}: 拒绝 bool（整数参数不得为布尔值）")
    if not isinstance(value, int):
        raise InvalidCanvasError(
            f"{name}: 必须是 int，得到 {type(value).__name__}={value!r}")
    return value


def _validate_canvas(width: int, height: int) -> None:
    w = _is_int(width, "width")
    h = _is_int(height, "height")
    if not 1 <= w <= MAX_CANVAS:
        raise InvalidCanvasError(
            f"width: 越界 {w}，允许范围 1..{MAX_CANVAS}")
    if not 1 <= h <= MAX_CANVAS:
        raise InvalidCanvasError(
            f"height: 越界 {h}，允许范围 1..{MAX_CANVAS}")


def _fraction_square_root(q: Fraction) -> Fraction:
    return Fraction(math.isqrt(q.numerator), math.isqrt(q.denominator))


def _is_square(q: Fraction) -> bool:
    r = math.isqrt(q.numerator)
    if r * r != q.numerator:
        return False
    r = math.isqrt(q.denominator)
    return r * r == q.denominator


def _sign(value: Fraction) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _sign_radical(a: Fraction, k: Fraction, d: Fraction) -> int:
    """精确计算 sign(a + k*sqrt(d))，要求 d > 0。"""
    if k == 0:
        return _sign(a)
    kkdd = k * k * d
    if k > 0:
        # a >= 0 时严格为正（sqrt(d) > 0）。
        if a >= 0:
            return 1
        return _sign(kkdd - a * a)
    if a <= 0:
        return -1
    return _sign(a * a - kkdd)


def _line_hit(seg: LineSegment, ys: Fraction):
    """直线命中行时返回 (0, dir, x:Fraction)。"""
    xa, ya = seg.p0
    xb, yb = seg.p1
    dy = yb - ya
    t = (ys - ya) / dy
    return (0, 1 if dy > 0 else -1, xa + t * (xb - xa))


def _piece_hit(p: MonoPiece, ys: Fraction):
    """单调曲线段命中行时返回行交点描述（见模块字符串），否则 None。"""
    yl = p.y0
    e = yl - ys
    pp, q = p.c2, p.c1
    direction = 1 if p.upward else -1

    if pp == 0:
        # y 线性；能命中则 q != 0（水平退化段区间为空，早已被排除）。
        t = (ys - yl) / q
        return (0, direction, p.a2 * t * t + p.a1 * t + p.x0)

    d = q * q - 4 * pp * e
    if d == 0:
        t = -q / (2 * pp)
        return (0, direction, p.a2 * t * t + p.a1 * t + p.x0)

    if _is_square(d):
        root = _fraction_square_root(d)
        t_hit = None
        for s in (1, -1):
            t = (-q + s * root) / (2 * pp)
            if 0 <= t <= 1:
                t_hit = t
                break
        # 半开区间保证命中时恰有一个根在 [0,1]。
        if t_hit is None:
            return None
        return (0, direction, p.a2 * t_hit * t_hit + p.a1 * t_hit + p.x0)

    # 无理根：不解根，直接在 Q(√D) 内为每个候选根做“t∈(0,1)”筛选，
    # 并预计算 x 比较的 (R0, M)；开区间定号用根的代数共轭表达式：
    # t_s<1 ⇔ s(q²-2pe) < s² q√D = q√D（两边乘 s>0 情形需分别处理，
    # 故直接对 2p 与 (-q+s√D) 差的符号做定号，见下 _root_inside）。
    for s in (1, -1):
        if _root_inside_open(pp, q, d, s):
            r0 = (2 * pp * pp * p.x0 - p.a1 * pp * q
                  + p.a2 * q * q - 2 * pp * p.a2 * e)
            m_rad = s * (p.a1 * pp - p.a2 * q)
            return (1, direction, pp, r0, m_rad, d)
    return None


def _root_inside_open(pp: Fraction, q: Fraction, d: Fraction, s: int) -> bool:
    """判定根 t_s=(-q+s√d)/(2p) 是否严格位于 (0,1)（d>0 非平方）。"""
    # sign(t)=sign(2p)·sign(-q+s√d)；
    # sign(1-t)=sign(2p)·sign(2p+q-s√d)。两者都需为正。
    sp = 1 if pp > 0 else -1
    sign_num = _sign_radical(-q, Fraction(s), d)
    sign_one_minus = _sign_radical(2 * pp + q, Fraction(-s), d)
    return sign_num == sp and sign_one_minus == sp


class RasterResult:
    """一次栅格化的不可变结果。

    属性：
        width, height: int
        coverage: ``tuple[tuple[float,...], ...]``，行优先，值为 k/16。
    """

    __slots__ = ("width", "height", "_winding", "coverage")

    def __init__(self, width: int, height: int,
                 winding: Tuple[Tuple[Tuple[Tuple[int, ...], ...], ...], ...],
                 coverage: Tuple[Tuple[float, ...], ...]):
        self.width = width
        self.height = height
        self._winding = winding
        self.coverage = coverage

    def coverage_at(self, x: int, y: int) -> float:
        """像素 (x,y) 的灰度覆盖（0.0..1.0，k/16 精确）。"""
        self._check_pixel(x, y)
        return self.coverage[y][x]

    def winding_at(self, x: int, y: int, sx: int, sy: int) -> int:
        """查询像素 (x,y) 内子样本 (sx,sy) 的精确环绕数。

        sx/sy 取 0..3；非零表示该子样本落在非零环绕区域。
        """
        self._check_pixel(x, y)
        sx = _sub_index(sx, "sx")
        sy = _sub_index(sy, "sy")
        return self._winding[y][x][sy][sx]

    def _check_pixel(self, x: int, y: int) -> None:
        xi = _query_int(x, "x")
        yi = _query_int(y, "y")
        if not 0 <= xi < self.width:
            raise InvalidQueryError(
                f"x: 像素下标 {xi} 越界（width={self.width}）")
        if not 0 <= yi < self.height:
            raise InvalidQueryError(
                f"y: 像素下标 {yi} 越界（height={self.height}）")


def _query_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise InvalidQueryError(f"{name}: 拒绝 bool")
    if not isinstance(value, int):
        raise InvalidQueryError(
            f"{name}: 必须是 int，得到 {type(value).__name__}")
    return value


def _sub_index(value, name: str) -> int:
    v = _query_int(value, name)
    if not 0 <= v < SUBSAMPLES:
        raise InvalidQueryError(
            f"{name}: 子样本下标 {v} 越界（允许 0..{SUBSAMPLES - 1}）")
    return v


def rasterize(glyph: GlyphOutline, width: int, height: int) -> RasterResult:
    """对校验过的轮廓做 4x4 子采样非零环绕栅格化。

    纯函数：不修改输入，失败只抛异常、不返回部分结果。
    """
    if not isinstance(glyph, GlyphOutline):
        raise InvalidCanvasError(
            "glyph: 必须由 outline() 构造的 GlyphOutline，得到 "
            f"{type(glyph).__name__}")
    _validate_canvas(width, height)

    lines, pieces = segments_of(glyph.contours)

    cols = SUBSAMPLES * width          # 全行采样列数
    rows_total = SUBSAMPLES * height   # 全行采样行数
    # grid[i][j]：全局子样本行 i=4py+sy、列 j=4px+sx 的环绕数。
    grid: List[List[int]] = [[0] * cols for _ in range(rows_total)]

    for py in range(height):
        for sy in range(SUBSAMPLES):
            i = py * SUBSAMPLES + sy
            ys = Fraction(2 * i + 1, 8)

            hits: list = []
            for seg in lines:
                if line_hits_row(seg, ys):
                    hits.append(_line_hit(seg, ys))
            for piece in pieces:
                if piece_hits_row(piece, ys):
                    hit = _piece_hit(piece, ys)
                    if hit is not None:
                        hits.append(hit)

            if not hits:
                continue

            # 射线向右：交点对全部满足 x_j < x_hit 的全局采样列 j 贡献
            # dir。j 与交点位置的序关系单调，用精确谓词二分求列上界
            # jmax（严格小于），再以差分数组累加：每行每交点仅 log 次定号。
            diff = [0] * (cols + 1)
            for hit in hits:
                jmax = _columns_left_of(hit, cols)
                diff[0] += hit[1]
                diff[jmax] -= hit[1]
            running = 0
            row_cells = grid[i]
            for j in range(cols):
                running += diff[j]
                row_cells[j] = running

    coverage = tuple(
        tuple(
            sum(1 for sy in range(SUBSAMPLES) for sx in range(SUBSAMPLES)
                if grid[py * SUBSAMPLES + sy][px * SUBSAMPLES + sx] != 0)
            / (SUBSAMPLES * SUBSAMPLES)
            for px in range(width)
        )
        for py in range(height)
    )
    winding_immutable = tuple(
        tuple(
            tuple(tuple(grid[py * SUBSAMPLES + sy][px * SUBSAMPLES:(px + 1) * SUBSAMPLES])
                  for sy in range(SUBSAMPLES))
            for px in range(width)
        )
        for py in range(height)
    )
    return RasterResult(width, height, winding_immutable, coverage)


def _columns_left_of(hit, cols: int) -> int:
    """满足采样列 x_j=(2j+1)/8 严格小于交点 x 的列数 jmax（0..cols）。

    谓词随 j 单调（真→假），二分上界；比较全程精确定号。
    """
    lo, hi = 0, cols
    while lo < hi:
        mid = (lo + hi) // 2
        xj = Fraction(2 * mid + 1, 8)
        if _hit_strictly_greater(hit, xj):
            lo = mid + 1
        else:
            hi = mid
    return lo


def _hit_strictly_greater(hit, xs: Fraction) -> bool:
    """交点 x 是否严格大于 xs（精确定号）。"""
    if hit[0] == 0:
        return hit[2] > xs
    _, _, pp, r0, m_rad, d = hit
    # 4p²(x_hit-xs)=2[(R0-2p²xs)+M√d]，4p²>0：
    return _sign_radical(r0 - 2 * pp * pp * xs, m_rad, d) > 0
