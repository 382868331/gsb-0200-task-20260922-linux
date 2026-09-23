# nonzero_fill：多边形字体轮廓的非零环绕填充

离线字体栅格内核的可复用模块：输入由**线段**与**二次贝塞尔曲线**组成的至多
20 条闭合轮廓（有理坐标），按 **nonzero winding（非零环绕）** 规则，以每像素
固定 **4×4 子采样**计算灰度覆盖，并支持查询每个子样本的精确环绕数。

全程只用 Python 标准库；**不调用任何字体库、绘图库或第三方同类实现**，
核心算法（曲线拆分、交点定号、环绕累加）全部在本仓库内从零实现。

## 运行环境

* Python **3.14.7**（代码不使用 3.14 之外的特性，仅在 3.14.7 上验证）
* Linux 原生；**无第三方依赖**，安装 Python 后即可运行
* 完全离线：不需要网络、账号、密钥、硬件、数据库、中间件、云 API 或 Docker

## 运行命令

```bash
# 必要测试（31 个用例）
python -m unittest discover -s tests -v

# 固定输入演示：真实计算并打印“洞 + 重叠”正常结果与三个实际触发的拒绝边界
python demo.py
```

实测耗时：整套测试约 5 秒（含 14 个随机二次曲线用例对独立参考的全网格比对），
`demo.py` 小于 0.1 秒；均不使用 sleep 或预录结果。

## 公开接口

```python
from nonzero_fill import point, line, quad, outline, rasterize

p0 = point(0, 0)                       # 有理坐标点（int 或 fractions.Fraction）
p1 = point(Fraction(1, 2), 3)
s1 = line(p0, p1)                      # 直线段 p0 -> p1
s2 = quad(p0, p1, point(2, 0))         # 二次贝塞尔 p0 -> p2，控制点 p1

glyph = outline([[s1, s2, ...], ...])  # 每条轮廓按段顺序首尾衔接并闭合
result = rasterize(glyph, width=16, height=16)

result.coverage                        # (height, width) 的 float，值为 k/16
result.coverage_at(x, y)               # 像素 (x,y) 覆盖，0.0..1.0
result.winding_at(x, y, sx, sy)        # 像素内子样本 (sx,sy)∈0..3 的精确环绕数 int
```

坐标约定：x 向右、y 向上；画布像素 `(x,y)` 覆盖正方形 `[x,x+1)×[y,y+1)`。
4×4 子样本位于 `(x+(2sx+1)/8, y+(2sy+1)/8)`（即像素的 1/8、3/8、5/8、7/8 处）。

### 数据结构

| 名称 | 含义 |
| --- | --- |
| `point(x, y)` | 校验并返回 `(Fraction, Fraction)`；拒绝 float/bool/NaN/Infinity |
| `line(p0, p1)` | `LineSegment`，拒绝零长度退化段 |
| `quad(p0, p1, p2)` | `QuadraticSegment`（二次贝塞尔，p1 为控制点），拒绝三点全同 |
| `outline(contours)` | `GlyphOutline`，原子校验：0..20 条轮廓，每段首尾衔接、末段闭合 |
| `rasterize(glyph, w, h)` | 纯函数，返回不可变 `RasterResult`；w/h 为 1..64 的 int |

### 异常（`nonzero_fill.errors`）

| 异常 | 触发情形 |
| --- | --- |
| `InvalidCoordinateError`（TypeError） | 坐标为 float（含 NaN/Infinity 点名）、bool 或无法识别类型 |
| `InvalidContourError`（ValueError） | 轮廓数 >20、空轮廓、段类型错误、端点不衔接、未闭合、退化段 |
| `InvalidCanvasError`（ValueError） | width/height 非 int、为 bool、超出 1..64；glyph 类型错误 |
| `InvalidQueryError`（IndexError） | 像素/子样本下标越界或类型错误（同样拒绝 bool） |

所有错误消息都带可定位前缀（如 `contours[3].segments[7]->segments[0]: 端点不衔接`）。
构造与栅格化都是原子操作：失败即抛异常，不返回对象、不产生部分结果。

## 算法与设计取舍

1. **全程有理精确。** 坐标统一为 `fractions.Fraction`，不引入任何浮点近似
   参与“内/外”判定。直线交点恒为 Fraction。
2. **二次曲线先拆 y 单调段。** 对 `y(t)=c t²+b t+y0` 在内部极值参数
   `t*=-b/(2c)` 处用 de Casteljau 精确拆分；拆后每段对任一水平线至多一个交点。
3. **半开 y 规则（标准 winding 约定）。** 每个交点按轮廓走向带方向：
   向上 +1、向下 -1；几何 y 区间统一为“含低端、不含高端”
   （`y_lo <= ys < y_hi`），与方向无关；水平段不计。顶点恰落采样行时，
   相邻两边合计恰为一次穿越；局部最低/最高（相切触碰）处合计为 0，
   等于把采样线微扰到顶点上方的极限，无需任何顶点特判。
4. **射线只向右，交点严格 `x_hit > xs` 才计入**（等号不计），
   `winding = Σ dir·[x_hit>xs]`；覆盖 = 16 个子样本中 winding≠0 的比例。
5. **二次无理根不消根、不近似。** 判别式 `D` 为完全平方时根是 Fraction；
   否则根落在二次扩域 `Q(√D)`。所有“交点 x 与采样列谁大”的比较都通过
   `sign(A+k√D)` 的精确代数定号完成（至多一次整数平方比较，`math.isqrt`）。
   因此本实现**没有“搜索/迭代预算耗尽”这一状态**：是否存在交点由半开区间
   严格判定，定号是精确且必然终止的；“证明无交点”（区间外/判别式为负）
   与“算不出来”在本算法里不可能混淆——后者根本不存在。
6. **性能。** 每条采样行把交点对采样列的贡献用“精确定号二分 + 差分数组”
   累加（每个交点仅 O(log(4W)) 次 Fraction 比较），而不是逐子样本重复求交。
   实测 20 条二次贝塞尔轮廓、64×64 上限画布约 0.7 秒；20 条直线多边形约 0.2 秒。

### 误差口径（重要）

* **所有内/外判定与环绕数在数学上精确**：Fraction 与 `Q(√D)` 定号无舍入，
  不存在用于“盖住边界错误”的容差；边界两侧的结果不同是半开规则的确定语义。
* **输出覆盖值零浮点舍入误差**：覆盖恒为 `k/16`（k=0..16），分母是 2 的幂，
  在二进制浮点中精确可表示（测试断言 `coverage*16` 往返为整数）。
* 唯一无法被任何离散采样回答的，是曲线**恰好穿过某子样本点**本身——本实现
  依据上述明确规则（严格 `x>xs` + 半开 y）给出确定答案，并由测试锁定，
  而不是靠容差把边界点“吸”到某一侧。

## 输入上限（接口边界）

* 轮廓数：**0..20**（0 条 = 空字形，合法；含空轮廓 `[[]]` 非法）
* 每轮廓：≥1 个段，段总数无额外限制（受闭合/衔接约束）
* 画布：width、height 均为 **1..64** 的 int
* 坐标：任意有界有理数（int / Fraction），允许负坐标与画布外坐标
* 子采样：固定 4×4（`SUBSAMPLES=4`，不可配置）

## 已知限制 / 非目标

* 只实现本题契约的数学子集：**直线段 + 二次贝塞尔**，不含三次及更高阶曲线、
  不解析任何字体文件格式（无 TTF/OTF/CFF/glyf 解析），也不宣称行业标准
  （OpenType/ScanConverter 等）级兼容。
* 不做抗锯齿之外的处理：无 hinting、无网格对齐、无 subpixel LCD 排列、
  无 EVEN-ODD 规则（仅 nonzero）。
* 覆盖量化固定为 17 个等级（k/16）；不输出连续覆盖率或距离场。
* 仅通过 `outline()` 接受已校验模型，不直接吃裸元组（裸元组会被带位置拒绝）。

## 目录

```
nonzero_fill/          本题库（可导入）
  errors.py            异常层次与错误语义
  geometry.py          段模型、二次曲线 y 单调拆分、半开规则
  model.py             point/line/quad/outline 与全部输入校验
  rasterizer.py        Q(√D) 精确定号、4×4 子采样、环绕累加、RasterResult
tests/
  test_nonzero_fill.py 31 个契约测试
  _reference.py        独立参考（Decimal 60 位 + Fraction tie 判定），
                       不调用被测核心的任何算法
demo.py                固定输入演示
```

测试期望值来源：直边多边形为测试内解析推导；二次曲线用例与独立 Decimal
参考全网格比对（独立参考不复用核心算法）；往返/对称性质（覆盖由环绕数重算、
整体反转环绕变号覆盖不变）仅作为补充断言。
