# winding_fill — 非零环绕多边形/二次曲线填充模块

为离线字体栅格内核提供的**可复用**模块：输入由线段与二次 Bézier 组成的至多
20 条闭合轮廓（有理坐标），在不大于 64×64 的画布上，按**每像素固定 4×4
子采样**、对水平射线用**半开 y 规则**累加带方向交点，以 **nonzero
winding（非零环绕）** 规则决定每个子样本是否被覆盖，输出 17 级灰度覆盖值
（k/16, k=0..16）并允许逐子样本查询有符号环绕数。

本轮新增**精确面积灰度入口**（fill_area）：像素值是单位方格内非零环绕
区域的**真实面积**，用 fractions.Fraction 有理数精确计算——不靠固定次数
超采样近似，绕两圈的区域也只算一次。该入口面向**多边形（直线段）轮廓**，
允许自交轮廓与相互重叠的多个轮廓，输出一个指定的小窗口（原点可为负）。

全部核心算法从零实现，**不调用任何字体/绘图库或第三方同类实现**。

## 运行环境

- Python **3.14.7**（标准库；3.10+ 的语法亦可运行，但以 3.14.7 为准）
- Linux 原生运行，**无需安装任何第三方依赖**，无需网络 / 账号 / 密钥 /
  硬件 / 数据库 / 中间件 / 云 API / Docker。

venv 准备（可选，仓库已带 .venv 符号链接时可直接用）：

```bash
python3.14 -m venv .venv
.venv/bin/python -m unittest discover -s tests -v
```

## 准确命令

```bash
# 必要测试（70 个），全部离线、确定性
python -m unittest discover -s tests -v

# 固定输入演示：4x4 子采样灰度（约 0.04 秒）
python demo.py

# 本轮演示：精确面积灰度的正常 + 边界 + 拒绝用例（约 0.05 秒，远低于 90 秒预算）
python demo_iteration.py
```

## 公开接口

```python
from winding_fill import (
    Point, Line, Quad, Contour,      # 不可变数据结构（frozen dataclass）
    build_contour,                  # 顶点链 -> Contour 的便捷构造器
    parse_contours,                 # 原始 dict 数据 -> Contour（严格校验）
    rasterize,                      # Contour 列表 -> CoverageResult
    rasterize_raw,                  # 校验 + 栅格化一步完成
    CoverageResult,                 # 结果对象
    fill_area,                      # 精确面积灰度新入口
    fill_area_raw,                  # 校验 + 精确面积一步完成
    AreaResult,                     # 精确面积结果对象
    InvalidInputError, WindingFillError,
    SUBSAMPLES,      # = 4（契约常量）
    MAX_CONTOURS,    # = 20
    MAX_CANVAS,      # = 64
    MAX_COORD_ABS,   # = 1024（允许的坐标绝对值上限）
)
```

### 数据结构

- `Point(x, y)`、`Line(p0, p1)`、`Quad(p0, c, p1)`（一个控制点的二次
  Bézier）、`Contour(segments)`，均为冻结 dataclass。
- 轮廓**几何闭合**：最后一段终点必须与第一段起点完全相等（按输入值精确
  比较）；相邻段必须首尾相接。没有独立的“方向”标志——方向完全由坐标顺序
  决定，反转整条轮廓即反转其环绕贡献。
- 原始数据格式（`parse_contours` / `rasterize_raw`）：

  ```python
  [                                 # 1..20 条轮廓
    [                               # 一条轮廓 = 段列表（>= 3 段）
      {"type": "line", "p0": {"x": 1, "y": 1}, "p1": {"x": 5, "y": 1}},
      {"type": "quad", "p0": {...}, "c": {"x": 3, "y": 0}, "p1": {...}},
      ...
    ],
  ]
  ```

- 便捷构造：`build_contour([(x0,y0), ...], quads=((边序号, cx, cy), ...))`
  自动把顶点链闭合（末点连回首点），可用 `quads=` 指定某条边为二次曲线。

### 结果查询

`result = rasterize(contours, width, height)` 返回 `CoverageResult`：

- `result.levels`：行主序元组 `levels[py][px] ∈ {0,...,16}`，被覆盖子样本
  数；`py=0` 为**底部**一行（坐标系 y 向上，像素 `(px,py)` 覆盖
  `[px,px+1]×[py,py+1]`）。
- `result.windings[py][px]`：16 个有符号环绕数，索引 `j*4+i`
  （`i`,`j` ∈ 0..3 为子样本列/行）。
- `coverage_at(px, py) -> float`：`k/16`。
- `windings_at(px, py) -> tuple[int,...]`：16 个环绕数。
- `winding_at(px, py, i, j) -> int`：单个子样本的有符号环绕数。

子样本 `(i,j)` 位于
`(px+(2i+1)/8, py+(2j+1)/8)`（像素中心 4×4 网格）。

### 精确面积灰度（本轮新增）

```python
res = fill_area(contours, left, bottom, width, height)
```

- 窗口覆盖像素 `(left .. left+width-1) x (bottom .. bottom+height-1)`；
  `left`/`bottom` 为整数像素坐标，**可以为负**；`width`/`height`
  为整数 1..64（bool 一律拒绝）。
- `res.areas[row][col]`：fractions.Fraction，像素单位方格内非零环绕
  区域的**精确面积**（0..1）；row 0 为窗口**底部**一行（y 向上）。
- `res.area_at(px, py) -> Fraction`：按**绝对像素坐标**查询。
- `res.gray_at(px, py) -> float`：同一面积的浮点灰度视图。
- `res.total_area() -> Fraction`：窗口内全部像素面积之和（精确）。
- 只接受**直线段**轮廓（多边形）：二次 Bézier 与像素边界的交点一般是
  无理数，不在精确有理契约内，传入即抛 InvalidInputError（可定位到
  具体段）。输入坐标可用 int/float/Fraction；Fraction 坐标全程保持
  精确，float 按其二进制精确值转换。
- 语义与 4x4 入口完全一致：非零环绕规则、半开 y 规则、点在边上算内部；
  边正好沿像素边界时只按面积计，**没有额外线宽**；自交、重叠、绕两圈
  都按非零环绕的数学定义处理（绕两圈不增亮度）。

## 输入上限与错误语义（接口边界）

| 项目 | 边界 |
|---|---|
| 轮廓数 | 1..**20** |
| 画布边长 | 整数 1..**64** |
| 每条轮廓段数 | ≥ 3（无上限，但坐标受下面限制） |
| 坐标绝对值 | ≤ **1024** |
| 数值 | int/float；**NaN、±Infinity 一律拒绝**；`bool` 不算数值、整数参数拒绝 `bool` |
| 结构 | 必须闭合、首尾相接；零长度直线、端点重合的二次曲线被拒绝（方向无定义） |

- 所有异常都继承 `WindingFillError`；非法输入抛 `InvalidInputError`
  （同时是 `ValueError`），错误消息带**可定位**路径，如
  `contours[2].segments[1].p0.x: NaN and Infinity are not accepted`
  （机器可读位置在 `exc.location`）。非法输入**不静默纠正**。
- 批量操作先完整校验、后计算；任一轮廓非法则整体抛错，不产生部分结果，
  不存在“已说明之外的部分变更”。
- 本模块没有“搜索/求解预算”这一类资源概念（几何求交为闭式/确定次数
  迭代），因此不存在“预算耗尽 vs 证明无解”的歧义；无解情形（如轮廓外）
  是正常结果 0，而不是错误。

## 算法与设计取舍

1. **y 单调拆分**：直线天然单调；二次 Bézier 的
   `y(t)=a t²+b t+y0` 在内驻点 `t*=-b/(2a)` 处拆成至多 2 段，使每段对 y
   严格单调或水平。
2. **半开 y 规则**：对每条子样本水平线 `sy`，单调段端点
   `lo<hi` 时按 `lo ≤ sy < hi` 计一次——共享顶点的两条邻边恰有一条计入，
   不重不漏；水平段显式不计。方向：上行 +1、下行 −1。
3. **二次曲线求交**：在单调段上解 `y(t)=sy`（求根公式），单调段保证根
   唯一；控制点与端点重合的退化二次曲线等价为直线，按直线精确处理。
4. **切点确定性**：曲线 y 极值由 `y(t*)` 一次算出，相邻两段共用同一
   float 极值：要么两段都计入（−1/+1 成对抵消），要么都不计入，不依赖
   容差。
5. **环绕累加**：交点按 x 排序后做有符号后缀和；子样本点的环绕数为其
   右侧（`x ≥ sx`，点恰在边上按内部处理）所有交点方向之和；`winding≠0`
   即覆盖。
6. **坐标系**：x 向右、**y 向上**，与字体 EM 网格习惯一致；渲染为图像时
   调用方自行翻转行序。

### 精确面积算法（fill_area）

1. **裁剪**：每条直线段用有理 Liang-Barsky 裁剪到像素单位方格，保留方向。
2. **切 slab**：像素内所有关键 x（裁剪后端点、边与边的精确交点）把像素
   切成竖直条带；条带内部没有边相交，边的上下顺序恒定。
3. **切面片**：每个条带被边曲线与像素上下边界切成水平面片；面片中点的
   环绕数用与 4x4 入口相同的半开 y 水平射线规则**精确**求得，非零面片
   贡献其梯形面积（线性边界下中点公式精确）。
4. 全部运算为 fractions.Fraction 精确有理运算：没有超采样近似，也没有
   人为容差；像素内无边时用**像素中心**判定内外（角点可能恰好在边上）。

### 浮点与误差口径（明确，不用宽容差掩盖边界）

- 覆盖级别 `levels` 与环绕数 `windings` 是**离散判定量**：测试对直线几何
  用 `fractions.Fraction` **精确有理**参考逐子样本比对（必须**完全相等**，
  不是“近似”）；对含二次曲线的几何用 60 位 `decimal.Decimal` +
  130 次二分求根（约 1e-39）的独立参考比对（**完全相等**）。参考实现
  位于 `tests/independent_reference.py`，**不导入被测核心**。
- 唯一使用 float 的地方是曲线求根的舍入：核心为 IEEE-754 双精度。测试
  坐标刻意取在 1/8 网格上，而 Decimal 参考以 ~1e-39 精度判定；当真实交点
  与某子样本 x 坐标的距离小于双精度舍入（~1e-15，乘以 ≤1024 的坐标量级）
  时，两侧判定一致。也就是说：**边界判定没有人为容差**；误差仅来自双精度
  求根本身（相对误差约 1e-15 量级），切点/顶点等退化情形由上面第 4 点的
  同一值共享机制消除二义性。
- `coverage_at` 是精确的 `k/16`（k 为整数 0..16），不产生其它浮点误差。

## 已知限制 / 非目标

- 仅实现本题契约的数学子集：线段 + 二次 Bézier；**不支持**三次及以上
  Bézier、圆弧、可变子采样率（固定 4×4）、子像素平移 hinting、
  anti-alias 滤波（覆盖率就是盒子滤波的 k/16）、奇偶(even-odd)规则。
- 精确面积入口（fill_area）进一步限定为**多边形（直线段）轮廓**；
  它面向**小窗口**输出设计，逐像素做精确排列求面积，复杂度约为
  每像素 O(段数²)，窗口很大时比 4x4 子采样入口慢（演示规模亚秒级）。
- 不宣称与任何行业标准（TrueType/OpenType scan-converter、SVG、
  D 算法细节、Porter-Duff 合成等）完整兼容；只对本文档描述的契约负责。
- 不做几何自交清洗：自交轮廓按 nonzero winding 的数学定义直接得到结果
  （测试含蝴蝶结自交用例）。
- 轮廓非常复杂时（段数很多）时间随“子样本行 × 段数”线性增长；上限
  规模（20 条曲边轮廓、64×64）实测亚秒级。

## 目录

```
winding_fill/__init__.py     # 公开 API
winding_fill/errors.py       # 异常类型
winding_fill/geometry.py     # 数据结构、严格校验、构造器
winding_fill/core.py         # 核心：单调拆分、半开 y、nonzero 栅格化
winding_fill/area.py         # 精确有理面积灰度（多边形轮廓 + 小窗口）
demo.py                      # 固定输入演示（洞 + 重叠 + 拒绝边界）
demo_iteration.py            # 本轮演示：精确面积正常/边界/拒绝用例
tests/test_validation.py     # 输入/错误语义/上限测试
tests/test_algorithm.py      # 与独立参考逐子样本对比及规则测试
tests/test_area.py           # 精确面积：手算对照 + 与独立参考的总面积核对
tests/independent_reference.py  # Fraction / Decimal / 整窗面积独立参考（不导入核心）
```

## 演示说明

`python demo.py` 真实计算一个 19×13 字形（CCW 曲边外框 + 与其重叠的 CCW
矩形 + CW 曲边洞），打印灰度图与 4 个固定子样本环绕数（普通区域 +1、
重叠区域 +2、洞内 0、外部 0），并实际触发两类拒绝：NaN 坐标
（定位到具体字段）与 `bool` 冒充整数画布宽度。无 sleep、无预录结果。
