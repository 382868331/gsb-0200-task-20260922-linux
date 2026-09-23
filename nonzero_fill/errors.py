"""非零环绕填充库的异常层次。

所有异常消息均带可定位信息（轮廓下标 contours[i]、段 segments[j]、
字段名或参数名），非法输入一律抛错，绝不静默纠正或猜测修正。
"""

from __future__ import annotations


class NonzeroFillError(Exception):
    """本库所有公开错误的基类。"""


class InvalidCoordinateError(NonzeroFillError, TypeError):
    """坐标分量非法。

    合法坐标为 ``int`` 或 ``fractions.Fraction``（:class:`numbers.Rational`）。
    特别地：``bool`` 被拒绝（即使它是 int 子类）；``float`` 被拒绝，
    且对 NaN / Infinity 给出明确点名的消息。
    """


class InvalidContourError(NonzeroFillError, ValueError):
    """轮廓结构非法。

    包括：轮廓数超过 20、空轮廓、未知段类型、相邻段端点不衔接、
    末段未闭合回首段、零长度退化段等。
    """


class InvalidCanvasError(NonzeroFillError, ValueError):
    """画布参数非法：width/height 非 1..64 的整数，或传入 bool。"""


class InvalidQueryError(NonzeroFillError, IndexError):
    """子样本/像素查询下标越界或类型错误。"""
