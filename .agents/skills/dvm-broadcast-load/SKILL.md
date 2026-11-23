---
name: dvm-broadcast-load
description: 为 PyTorch-NPU 生成 Ascend950 的 DVM 广播自定义算子。识别输入侧 View、Unsqueeze、Expand 加逐元素计算，把广播折到零 stride 的 view_load，利用 C310 NDDMA 减少 Vector Broadcast。
---

# Broadcast：在 GM→UB 搬运时展开输入

先找直接来自 GM 的广播输入，例如逐行 scale、逐通道 bias。若 DAS 中 `BroadcastX/Y` 后面紧接着 Mul/Add，可尝试让搬运单元按消费形状填充 UB，省去 Vector 广播指令。已经在 UB 中算出的中间结果适合保留原图；为使用这条规则而额外写回 GM 会多一次往返。

**构造地址映射。** 沿输入的 view、切片、permute、unsqueeze、expand 还原每个消费坐标对应的物理源元素。保留移动轴的真实元素 stride，对新增或由长度 1 扩展的轴设 stride=0。固定 slice 可用对应 Tensor 视图的起始地址；实参的 `data_ptr()` 已含 storage offset。只把存在仿射地址映射的部分移到 load，保留 cast、舍入和计算的语义顺序。重复地址只用于读。

## 例：连续 FP32 矩阵逐行缩放

目标 `y[r,t] = x[r,t] * scale[r]`。基线把 scale 声明为 `k.load([R,1], dvm.float32)`，由乘法触发广播。候选直接将逻辑元素 `[r,t]` 映射到源偏移 `r`。

```python
from torch_npu._inductor import dvm

R, T = 128, 256

@dvm.kernel(ktype="vector", dyn_shape=False)
def row_scale(k):
    x = k.load([R, T], dvm.float32)
    scale = k.view_load([R, T], [1, 0], dvm.float32)
    k.store(k.mul(x, scale))

# 静态调用传原始 R 个 scale，输出为 [R,T]
# row_scale.run(x, scale, out)
```

- 本机独立 DVM 的 `Ascend950DT_9581` host DAS 在这一形状上将 `load + BroadcastX + Mul` 变为 `view_load_x + Mul`。在目标分支沿 `src/ops.cc` 的 `V_LOAD_VIEW_X` 和 `src/vm_aiv_c310.cce` 的 `ViewLoadXImpl → nddma` 核实。普通 `view_load` 也可能选中其他 DMA handler；需要专用广播搬运时，确认实际 opcode、源 stride 和 handler。
- 对外轴广播，如 `bias[C] → [B,C]`，候选 stride 为 `[0,1]`。若原路径能复用小块数据，逐 tile 展开可能增加搬运或占用更多 UB。对比 Vector/MTE2 时间、tile 大小和 kernel 总耗时，按实测 shape 选版本。
- 动态 PTA 若从实参取 strides，将 `scale[:, None].expand(R,T)` 作为零拷贝视图传给对应的动态 ViewLoad；检查 wrapper 没有物化 expanded Tensor。对照 PyTorch 显式广播，覆盖各行不同的值、T=1、尾块和带 offset 的输入。测量应包含调用方为候选引入的任何额外操作。
