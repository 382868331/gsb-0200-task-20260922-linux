"""固定输入演示：非零环绕填充的“洞”与“重叠”，外加一个真实拒绝边界。

运行：python demo.py
全程离线、确定性坐标、无第三方依赖、无 sleep / 预录结果。
"""

from __future__ import annotations

from nonzero_fill import (
    InvalidContourError,
    InvalidCoordinateError,
    line,
    outline,
    point,
    quad,
    rasterize,
)

# k/16 覆盖 -> 终端字符（16 级灰度感）。
_GRAY = " .:-=+*#%@"


def render(result) -> str:
    """把覆盖矩阵画成字符图（终端上行号从上到下，故纵向翻转）。"""
    rows = []
    for py in range(result.height - 1, -1, -1):
        line_chars = []
        for px in range(result.width):
            k = round(result.coverage_at(px, py) * 16)
            line_chars.append(_GRAY[min(k, 9)])
        rows.append("".join(line_chars))
    return "\n".join(rows)


def ring_outline():
    """带洞的“O”：外叶形（二次贝塞尔）CW + 内叶形反向 CCW。"""
    # 外叶：W->N->E->S（y 向上坐标里为 CW，内部 winding=-1）。
    outer = [
        quad(point(1, 8), point(1, 15), point(8, 15)),
        quad(point(8, 15), point(15, 15), point(15, 8)),
        quad(point(15, 8), point(15, 1), point(8, 1)),
        quad(point(8, 1), point(1, 1), point(1, 8)),
    ]
    # 内叶反向（CCW）：W->S->E->N，环绕数 +1，与外叶在洞中抵消为 0。
    inner = [
        quad(point(4, 8), point(4, 5), point(8, 5)),
        quad(point(8, 5), point(12, 5), point(12, 8)),
        quad(point(12, 8), point(12, 11), point(8, 11)),
        quad(point(8, 11), point(4, 11), point(4, 8)),
    ]
    return outline([outer, inner])


def overlapping_outline():
    """两个同向重叠圆角块：相交区 winding=2，nonzero 规则下仍填充。"""
    a = [
        line(point(1, 1), point(9, 1)),
        line(point(9, 1), point(9, 9)),
        line(point(9, 9), point(1, 9)),
        line(point(1, 9), point(1, 1)),
    ]
    b = [
        line(point(6, 6), point(14, 6)),
        line(point(14, 6), point(14, 14)),
        line(point(14, 14), point(6, 14)),
        line(point(6, 14), point(6, 6)),
    ]
    return outline([a, b])


def main() -> None:
    print("=" * 60)
    print("演示 1：带洞的 O（外 CW 二次贝塞尔叶形 + 内反向叶形）")
    print("=" * 60)
    ring = ring_outline()
    r1 = rasterize(ring, 16, 16)
    print(render(r1))
    # 子样本环绕数可查询：实体环、洞内、画布外。
    print("\n子样本环绕数 winding_at(px, py, sx, sy)：")
    print("  外实体环 (1,8,s0,s0) =", r1.winding_at(1, 8, 0, 0))
    print("  洞内     (8,8,s0,s0) =", r1.winding_at(8, 8, 0, 0),
          "（外 -1 + 内 +1，精确抵消）")
    print("  画布外   (15,15,s3,s3) =", r1.winding_at(15, 15, 3, 3))
    print("  环像素覆盖 coverage_at(1,8) =", r1.coverage_at(1, 8))

    print()
    print("=" * 60)
    print("演示 2：两个同向重叠块（相交区 winding=2，仍被 nonzero 填充）")
    print("=" * 60)
    r2 = rasterize(overlapping_outline(), 16, 16)
    print(render(r2))
    print("\n子样本环绕数：")
    print("  仅 A 区 (2,2,s0,s0) =", r2.winding_at(2, 2, 0, 0))
    print("  相交区 (7,7,s0,s0) =", r2.winding_at(7, 7, 0, 0),
          "（两层同向环绕，nonzero 下非零 → 填充）")
    print("  仅 B 区 (12,12,s0,s0) =", r2.winding_at(12, 12, 0, 0))
    print("  外部    (0,13,s0,s0) =", r2.winding_at(0, 13, 0, 0))

    print()
    print("=" * 60)
    print("边界演示：非法输入被拒绝（可定位，不静默纠正）")
    print("=" * 60)
    # 真实触发 1：NaN 坐标。
    try:
        point(float("nan"), 0)
    except InvalidCoordinateError as exc:
        print(f"[拒绝] NaN 坐标 -> {type(exc).__name__}: {exc}")

    # 真实触发 2：21 条轮廓，超过 20 的接口上限。
    too_many = [
        [quad(point(0, 0), point(1, 1), point(0, 0))]  # 合法单段自闭合
        for _ in range(21)
    ]
    try:
        outline(too_many)
    except InvalidContourError as exc:
        print(f"[拒绝] 轮廓超限 -> {type(exc).__name__}: {exc}")

    # 真实触发 3：bool 不得作为整数参数。
    try:
        rasterize(outline([]), True, 8)
    except Exception as exc:  # noqa: BLE001 - 演示需打印具体类型
        print(f"[拒绝] bool 画布宽 -> {type(exc).__name__}: {exc}")

    print("\n演示完成：以上结果均为本次真实计算（Fraction/Q(√D) 精确定号）。")


if __name__ == "__main__":
    main()
