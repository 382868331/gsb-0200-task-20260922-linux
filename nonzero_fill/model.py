"""公开数据结构、构造器与输入校验。

接口（坐标一律使用有理数；推荐 ``int`` 或 ``fractions.Fraction``）::

    pt  = point(x, y)
    seg = line(p0, p1)
    seg = quad(p0, p1, p2)                 # 二次贝塞尔，控制点 p1
    glyph = outline([[seg, ...], ...])     # 每条轮廓按段顺序首尾衔接并闭合
    result = rasterize(glyph, width, height)

校验在构造期一次性完成：任何非法输入都带可定位信息抛错（见 :mod:`errors`），
失败时不返回对象、不产生部分结果。

数值规则：

* 坐标只接受 ``numbers.Rational``（``int`` / ``Fraction``）；``float`` 一律拒绝，
  其中 NaN / Infinity 会被点名；``bool`` 即使是 int 子类也显式拒绝。
* 轮廓数 0..20；每条轮廓至少 1 段；相邻段必须端点衔接，末段必须回到首段起点。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from numbers import Rational
from typing import Iterable, Sequence, Tuple

from .errors import InvalidContourError, InvalidCoordinateError
from .geometry import LineSegment, QuadraticSegment, Segment

Point = Tuple[Fraction, Fraction]

MAX_CONTOURS = 20


def _coord(value, where: str) -> Fraction:
    """校验并转换单个坐标分量。"""
    if isinstance(value, bool):
        raise InvalidCoordinateError(f"{where}: 拒绝 bool（整数参数不得为布尔值）")
    if isinstance(value, float):
        if math.isnan(value):
            raise InvalidCoordinateError(f"{where}: 拒绝 NaN")
        if math.isinf(value):
            raise InvalidCoordinateError(f"{where}: 拒绝 Infinity（±inf）")
        raise InvalidCoordinateError(
            f"{where}: 拒绝 float {value!r}，坐标必须是有理数 int/Fraction")
    if isinstance(value, Rational):
        return Fraction(value.numerator, value.denominator)
    raise InvalidCoordinateError(
        f"{where}: 无法识别的坐标类型 {type(value).__name__}={value!r}，"
        "需要 int 或 Fraction")


def point(x, y) -> Point:
    """构造一个有理坐标点 ``(Fraction, Fraction)`` 并做数值校验。"""
    return (_coord(x, f"point.x({x!r})"), _coord(y, f"point.y({y!r})"))


def _as_point(value, where: str) -> Point:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise InvalidCoordinateError(
            f"{where}: 点必须是长度为 2 的序列 (x, y)，得到 {value!r}")
    return (_coord(value[0], f"{where}.x"), _coord(value[1], f"{where}.y"))


def line(p0, p1) -> LineSegment:
    """构造直线段 ``p0 -> p1``（拒绝零长度退化段）。"""
    a = _as_point(p0, "line.p0")
    b = _as_point(p1, "line.p1")
    if a == b:
        raise InvalidContourError(f"line: 零长度退化段 p0=p1={a}")
    return LineSegment(a, b)


def quad(p0, p1, p2) -> QuadraticSegment:
    """构造二次贝塞尔段 ``p0 -> p2``（控制点 p1）。

    三点全同的退化曲线被拒绝；``p0 == p2`` 的单段自闭合曲线合法。
    """
    a = _as_point(p0, "quad.p0")
    c = _as_point(p1, "quad.p1")
    b = _as_point(p2, "quad.p2")
    if a == c == b:
        raise InvalidContourError(f"quad: 三点全同的退化曲线 {a}")
    return QuadraticSegment(a, c, b)


@dataclass(frozen=True)
class GlyphOutline:
    """校验通过的字形轮廓：不可变、坐标全为 Fraction。"""

    contours: Tuple[Tuple[Segment, ...], ...]

    @property
    def contour_count(self) -> int:
        return len(self.contours)


def _seg_start(seg: Segment) -> Point:
    return seg.p0


def _seg_end(seg: Segment) -> Point:
    return seg.p2 if isinstance(seg, QuadraticSegment) else seg.p1


def outline(contours: Iterable[Iterable[Segment]]) -> GlyphOutline:
    """校验并构造 :class:`GlyphOutline`。

    ``contours`` 为段列表的列表；每个段必须由 :func:`line` / :func:`quad` 构造。
    所有检查通过后才构造对象（失败无部分产出）。
    """
    if isinstance(contours, (str, bytes)) or not isinstance(
            contours, (tuple, list)):
        raise InvalidContourError(
            "contours: 必须是轮廓的列表/元组（至多 "
            f"{MAX_CONTOURS} 条），得到 {type(contours).__name__}")
    if len(contours) > MAX_CONTOURS:
        raise InvalidContourError(
            f"contours: 轮廓数 {len(contours)} 超过上限 {MAX_CONTOURS}")

    built: list[tuple[Segment, ...]] = []
    for ci, contour in enumerate(contours):
        where = f"contours[{ci}]"
        if isinstance(contour, (str, bytes)) or not isinstance(
                contour, (tuple, list)):
            raise InvalidContourError(
                f"{where}: 轮廓必须是段的列表/元组，得到 "
                f"{type(contour).__name__}")
        if len(contour) == 0:
            raise InvalidContourError(f"{where}: 空轮廓（至少需要 1 个段）")

        segs: list[Segment] = []
        for si, seg in enumerate(contour):
            sw = f"{where}.segments[{si}]"
            if not isinstance(seg, (LineSegment, QuadraticSegment)):
                raise InvalidContourError(
                    f"{sw}: 段必须由 line()/quad() 构造，得到 "
                    f"{type(seg).__name__}")
            segs.append(seg)

        # 衔接性与闭合性。
        for si in range(len(segs)):
            cur = segs[si]
            nxt = segs[(si + 1) % len(segs)]
            end = _seg_end(cur)
            start_next = _seg_start(nxt)
            if end != start_next:
                raise InvalidContourError(
                    f"{where}.segments[{si}]->segments[{(si + 1) % len(segs)}]:"
                    f" 端点不衔接 {end} != {start_next}；轮廓必须首尾衔接并闭合")
        built.append(tuple(segs))

    return GlyphOutline(tuple(built))
