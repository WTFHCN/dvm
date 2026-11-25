---
name: dvm-fusion-support
description: 查询 Ascend950DT / C310 上 DVM 算子的直接融合和输入输出 dtype，选择 KernelType 和 KernelFlag，并对照 C++ Reset 与独立 DVM Python 写法。
---

# DVM 融合能力查询

按子图查直接融合矩阵，按参数角色查 dtype 表。选择执行方式时，给出 kernel 的用途、C++ 枚举和 Python 名称。以下结果以 Ascend950DT / C310 的 [DVM 版本 494ca9a](https://github.com/mindspore-ai/dvm/tree/494ca9a469a83facd279e4f390e490b9aba50df2) 为基准。

## 直接融合

✓ 表示至少一种常规静态形态能在同一个 `kVector` 或 `kMix` 中完成，× 表示该路径不支持直接组合。`kSplit` 自动拆图、推测分段和跨 stage 交换均不计入。ViewSch 列指由计算结果 Slice 或 Concat 触发的视图调度，VF 列指该算子本体能否编入 VfFusion 生成的函数，Fractal 列要求实际生成转置读取。

| DVM 特性 | 逐元素同核 | Reduce 同核 | ViewSch | CV 融合 | VF 函数内 | Fractal 转置 | OneHot 同核 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GatherLoad | ✓ | ✓ | × | × | × | × | ✓ |
| Reduce（普通 Sum） | ✓ | ✓ | ✓ | × | × | ✓ | ✓ |
| 逐元素（Add / Abs） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Cast | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| BroadcastTo（旁路广播） | ✓ | ✓ | ✓ | ✓ | × | ✓ | ✓ |
| ViewLoad（转置 stride） | ✓ | ✓ | ✓ | × | × | ✓ | ✓ |
| Slice（计算结果切片） | ✓ | ✓ | ✓ | × | × | × | ✓ |
| Concat | ✓ | ✓ | ✓ | × | × | × | ✓ |
| CV 融合（MatMul 端） | ✓ | × | × | ✓ | × | × | × |
| OneHot | ✓ | ✓ | ✓ | × | × | ✓ | ✓ |

- **CV 后处理**。一个 MatMul 可以接 Cast 和输出域上的逐元素计算，旁路张量可以先广播。普通 Reduce 和会增加类别轴的 OneHot 均不属于这类后处理。CV 行的 VF 单元格指 MatMul 本体。[Mix 规格](../../../docs/tutorial.md#L230-L247) · [Mix 构图](../../../src/xkernel.cc#L157-L190)
- **GatherLoad 的寻址**。GatherLoad → Slice 的主机对照中，切片起点从第 0 行改为第 2 行后，Gather 的 DAS 相同，普通 Load 对照的读取偏移相差 256 字节。GatherLoad 发码还要求没有 CV 分片。转置 stride 的旁路 ViewLoad 会直接进入 EmitView，绕过普通 Load 的 CV 分片寻址。[GatherLoad 发码](../../../src/ops.cc#L1188-L1219) · [Load 分片寻址](../../../src/ops.cc#L769-L807) · [View 访问准入](../../../src/ops.cc#L658-L672)
- **归约与视图**。普通逐元素 → FP32 Sum 可同核，并列归约需要兼容的输出域。Concat → Sum 有仓内用例。Slice → Concat ✓，Concat → Slice ×，同一通用 ViewSch 内的 Concat → Concat ×。计算结果 Slice、Concat 所用的 ViewSch 与 Fractal 转置走不同的调度分支。[Concat → Reduce 用例](../../../tests/ut/test_view_sch.py#L402-L415) · [ViewSch 拓扑限制](../../../src/schedule.cc#L1094-L1112) · [调度选择](../../../src/kernel.cc#L1568-L1575)
- **OneHot 的数据流方向**。int32 索引 `[64,128]`，`depth=8`，`axis=-1`，FP32 输出，可以直接接逐元素乘法或末轴 Sum。沿索引的行轴切去前两行，主机 DAS 的读取偏移是 1024 字节，类别深度仍是 8。两路同形 OneHot 也能同核相加。先分别对 `[64,128]` 和 `[23,128]` 编码再拼行时，两路保留深度 8。先拼接索引再做 OneHot 时，所查形状发出的类别深度为 112/120，该方向在这个例子中为 ×。[OneHot 组合用例](../../../tests/ut/test_onehot.py#L50-L115) · [OneHot 深度编码](../../../src/ops.cc#L3392-L3427)
- **OneHot 与 Fractal**。✓ 的正例是 `OneHot(indices[64], depth=128, axis=-1) + side.T`，转置后的加数为 `[64,128]`。同一段生成 `OneHot.b32`、`view_load_t` 和 `Add.fp32`。若直接转置 `[64,128]` 的索引并在最前插入类别轴（`depth=8, axis=0`），主机生成 `MODE_Y_TILE, dup_round=0`，而设备实现用该字段取模，该形态为 ×。[Fractal 准入](../../../src/schedule.cc#L216-L260) · [OneHot 设备实现](../../../src/vm_aiv_c310.cce#L1428-L1464)
- **VF 的边界**。OneHot 在 C310 上调用专用 SIMT 函数。VfFusion 只选择候选表内的算子，OneHot、Reduce 和访存节点可以留在同一 Vector kernel 的 VF 函数外。输入或输出含 int64 的节点被 VF 候选检查排除。[VF 候选判定](../../../src/vf_fusion.cc#L95-L149)

## 算子与 dtype

✓ 表示该参数角色支持此类型，包括 builder 在同一 kernel 内插入转换的情况。二元运算按两个同类型 Tensor 判断。FP16、BF16、FP32 分别指 float16、bfloat16、float32。[公开类型](../../../include/dvm.h#L26-L35) · [Vector 指令表](../../../src/ops.cc#L28-L108)

| 算子及参数角色 | bool | FP16 | BF16 | FP32 | int32 | int64 |
| --- | --- | --- | --- | --- | --- | --- |
| Add / Sub 数据 | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| Mul / Maximum / Minimum 数据 | × | ✓ | ✓ | ✓ | ✓ | × |
| Abs 输入 | × | ✓ | ✓ | ✓ | ✓ | × |
| Sqrt / Log / Exp 输入 | × | ✓ | ✓ | ✓ | × | × |
| Div / Pow 数据 | × | ✓ | ✓ | ✓ | × | × |
| Round / Floor / Ceil / Trunc 输入 | × | ✓ | ✓ | ✓ | × | × |
| Equal / NotEqual / Less / LessEqual / Greater / GreaterEqual 的比较输入 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| ReduceSum 输入 | × | × | × | ✓ | × | × |
| ReduceMax / ReduceMin 输入 | × | ✓ | × | ✓ | × | × |
| BroadcastTo 张量数据 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| GatherLoad 的源数据和输出 | × | ✓ | ✓ | ✓ | ✓ | × |
| GatherLoad 的索引 | × | × | × | × | ✓ | × |
| OneHot 的索引 | × | × | × | × | ✓ | × |
| OneHot 的 on/off 值及输出 | × | ✓ | ✓ | ✓ | ✓ | × |
| MatMul 的 A/B 输入 | × | ✓ | ✓ | × | × | × |
| Cast 源类型和目标类型 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

- **OneHot 与 Sum**。OneHot 索引为 int32。C++ 的 `on_value` 和 `off_value` 模板类型决定输出 dtype，Python 的 `one_hot` 显式传输出类型。要直接接 Sum，可选 FP32 输出。需要保留低精度或 int32 的 OneHot 结果时，在归约分支显式 Cast 到 FP32。普通 Sum 对其他输入类型返回空指针。[OneHot 与 Reduce 入口](../../../src/dvm.cc#L1509-L1543)
- **逐元素与 Cast**。BF16 的 Abs、Sqrt、Log、Exp、Div，以及 FP16/BF16 的 Pow 和取整运算可先升至 FP32 再转回。比较结果为 bool。普通 Vector 的六种类型两两 Cast 均已做主机代码生成检查，包括相同类型的直接复用。FP16 → BF16 和 int32 → BF16 都经过 FP32。[一元提升](../../../src/dvm.cc#L1244-L1282) · [二元提升](../../../src/dvm.cc#L1320-L1396) · [Cast 构图](../../../src/dvm.cc#L1429-L1447)
- **GatherLoad 与 MatMul**。GatherLoad 通过 `GlobalAccess` 传 int32 索引，源数据按 16 或 32 位搬运，输出继承源类型。MatMul 的 A、B 使用同一种 FP16 或 BF16，普通结果节点继承 A 的类型。Cube 的 FP32 输出位与后续 Cast 分开判断。[GatherLoad 索引](../../../src/vm_aiv_c310.cce#L135-L163) · [MatMul 结果类型](../../../src/ops_m.cc#L200-L209) · [Cube 输出编码](../../../src/ops_m.cc#L675-L687)

## kernel 类型

C++ 通过 `dvm::Kernel::Reset(type, flags)` 选择类型。七种类型的 Python 名称适用于独立 DVM 包。[C++ 枚举](../../../include/dvm.h#L96-L117) · [Python 解析器](../../../src/pybind_api.cc#L117-L154)

| C++ KernelType | Python 类型名 | 什么时候选 | 执行方式 |
| --- | --- | --- | --- |
| `dvm::kVector` | `vector` | 逐元素、OneHot、GatherLoad 或普通归约。例如 FP32 输出的 OneHot 接 Sum。 | 在 AIV 上执行。直接融合要求计算域兼容，多个归约的输出 shape 需要一致。 |
| `dvm::kCube` | `cube` | 单个 MatMul、批量 MatMul 或 GroupedMatMul | 矩阵乘走 AIC。可使用该算子自身支持的转置和 bias。 |
| `dvm::kMix` | `mix` | 一个 MatMul 后接 Cast、bias Add 或逐元素激活 | AIC 做 Cube，AIV 按 Cube 输出 tile 做后处理。CV + Reduce ×，CV + OneHot ×。 |
| `dvm::kParallel` | `parallel` | 两条分支可以独立计算，例如一路做 MatMul，另一路对不同输入求和。 | 用 `ParallelAdd` 指定 Vector、Cube 或 Mix 子 kernel，分配到不同核并行执行，可设每个子 kernel 的核数上限。 |
| `dvm::kSequence` | `seq` | 后段依赖前段的结果，例如先做 MatMul，再由单独的 Vector 子 kernel 对结果求和。 | 用 `SequenceAdd` 显式指定子 kernel 的执行顺序，段间通过 Store/Load 传递数据。 |
| `dvm::kSplit` | `split` | 固定子图交给 DVM 自动划分，并可复用构图信息 | 依据算子和计算域拆成可执行子 kernel。 |
| `dvm::kEager` | `eager` | 上层按执行顺序逐步构图，每轮重新收集算子 | 按收集到的图分区执行，可选延迟发码或统一 workspace。 |

## kernel flags

常规配置传 `0`。C++ 多个位用按位或组合，Python 在类型名后加一个冒号，用英文逗号连接标志，例如 `"split:dyn,unify_ws"`。判断某个位是否生效要看构造工厂及对应 kernel 的实现。[构造分派](../../../src/dvm.cc#L1059-L1106)

| C++ KernelFlag | Python 后缀 | 生效范围和含义 |
| --- | --- | --- |
| `dvm::kDynamic`（`0x1`） | `dyn` | 图结构可复用，每次按实际输入 shape 重新推导形状并生成代码。Vector、Cube/Mix、Parallel、Sequence、Split 有动态形状路径。Parallel、Sequence 会把动态位传给子 kernel。Eager 自身按轮次重新构图。 |
| `dvm::kUnifyWS`（`0x2`） | `unify_ws` | 常规 Split、Eager 的跨段 workspace 可向上层申请一块内存，再按偏移绑定各临时槽。适合调用方只提供一块 workspace 的接口。 |
| `dvm::kSpeculate`（`0x4`） | `spec` | Vector 先尝试整体融合，shape 域或归约 tile 不满足条件时走备用执行路径。常规实现可用 `SpecNext()` 指定备用阶段的边界。检查实际生成的段数再判断是否同核。 |
| `dvm::kLazyCodeGen`（`0x8`） | `lazy` | Eager 在 CodeGen 阶段规划分区、地址与 workspace，把子 kernel 的具体发码延后到 Launch。可与 `unify_ws` 配合。 |
| `dvm::kOptFractalTrans`（`1u << 29`） | `opt_fractal` | 普通 Vector 选择 Fractal 转置调度，针对有合适 stride 的 2/4 字节 ViewLoad。没有合适的转置轴时可回到常规发码。是否命中看 DAS 的 `view_load_t`。这一构造分支与普通 ViewSch 择一使用。 |
| `dvm::kPrivate1`（`1u << 30`） | `priv1` | 内部路由位。在 `vector:spec,priv1` 选择另一套推测 Vector 实现，`SpecNext()` 在这一路径不划分阶段。在 `split:priv1` 选择 GraphKernel。语义依赖版本和 kernel 类型。 |
| `dvm::kPrivate2`（`1u << 31`） | `priv2` | 保留位。本版本 C++ 枚举和 Python 解析器识别该位，源码没有读取它来改变执行的分支。 |

统一 workspace 的实现会为临时槽分配偏移。Eager 的 lazy 路径会在 Launch 阶段生成各分区代码。VF 融合单独由 `Config::SetVfFusion` / `Kernel.set_vf_fusion` 配置。[workspace 分配](../../../src/xkernel.cc#L2802-L2837) · [Eager 延迟发码](../../../src/xkernel.cc#L3049-L3063) · [推测归约检查](../../../src/kernel.cc#L2329-L2399) · [SpecNext](../../../src/dvm.cc#L1721-L1725) · [VF 开关](../../../include/dvm_py.h#L229-L245)

### C++ 与 Python 写法

| 场景 | C++ | Python |
| --- | --- | --- |
| 普通静态 Vector | `k.Reset(dvm::kVector, 0)` | `Tester("vector")` |
| MatMul → Cast → Add | `k.Reset(dvm::kMix, 0)` | `Tester("mix")` |
| 转置 ViewLoad | `k.Reset(dvm::kVector, dvm::kOptFractalTrans)` | `Tester("vector:opt_fractal")` |
| 动态 shape、统一 workspace | `k.Reset(dvm::kSplit, dvm::kDynamic \| dvm::kUnifyWS)` | `Tester("split:dyn,unify_ws")` |
| Eager 延迟发码 | `k.Reset(dvm::kEager, dvm::kLazyCodeGen)` | `Tester("eager:lazy")` |

`k` 是 `dvm::Kernel` 实例，`Tester` 来自 `dvm.tester`。该版本中，`Tester()` 默认为 Vector，而独立包的 `@dvm.kernel` 默认为动态 Split。构造常规静态 Vector 时可显式写 `@dvm.kernel(ktype="vector", dynamic=False)`。[装饰器实现](../../../python/dvm/jit.py) · [Tester 构造](../../../python/dvm/tester.py#L70-L85)

## 在主机核对 OneHot → Sum

在已构建并配置为 C310 的 DVM 环境中，用 DAS 模式检查图和实际指令。

```python
import numpy as np
from dvm.tester import Tester

t = Tester("vector", use_pass_opt=True, run_mode="das")
indices = np.arange(64 * 128, dtype=np.int32).reshape(64, 128) % 8
encoded = t.one_hot(t.load(indices), 8, -1, 1.0, 0.0, "float32")
t.store(t.sum(encoded, [2], True))
t.codegen()
print(t.dump())
print(t.das())
```

该例应在一个 Vector 计算图中发出 `OneHot.b32` 和 `SumX.fp32`。遇到表外组合或目标版本变化，检查生成程序的阶段数、目标指令、数据类型，以及切片偏移或分块深度。表中的主机 DAS 证据只覆盖代码生成，设备数值需要单独对照。
