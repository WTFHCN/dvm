---
name: dvm-reduction-tiling
description: 为 PyTorch-NPU 生成 Ascend950/C310 的 DVM 自定义 Reduce 算子。用 view_load 重排逻辑访问轴、用 view_store 写回原布局，尝试减少或消除多核 Sum 的原子加。
---

# Reduce：用访存布局换掉部分结果写回

为目标算子生成两个等价 builder。基线按原布局 load、reduce、store。基线 DAS 若有 `store_atomic`、归并或额外清零，优先试一个 View 版本。Ascend950 的带宽和多维 View 搬运使非连续读值得尝试；选型要比较增加的搬运成本和省下的写回成本。

1. 从外部输入 Tensor 取真实 shape 和**元素 stride**。把保留轴放在一起，把归约轴放在一起，再用同一个置换重排 shape 和 stride。改用新轴编号做 reduce。这样有机会让每个核负责一组独立输出，在核内完成该组的归约。避免引入 GM 转置或 `.contiguous()`。
2. 输出逻辑轴若也换了顺序，把原输出 Tensor 的 stride 按保留轴的新顺序传给 `view_store`。内部保留单例归约轴时，给出同秩的写回 stride；外部可用连续的目标形状缓冲。若保留轴顺序和物理布局相同，可直接 store。
3. 看生成的 DAS 是否还把同一输出交给多个核。完整归约才能直接写最终结果；跨核 partial 仍需原子或归并。bias、ReLU 等后处理应在最终值上做。若基线已经没有原子写，先比较 View 是否仍改善总耗时。

## 可套用的静态 FP32 builder

外部连续输入 `x[B,P,C,H,W]`，目标为 `x.sum((0,3))`，输出顺序是 `[P,C,W]`。把逻辑读顺序改成 `[C,P,W,B,H]`，让 B、H 在内部形成归约域。写回偏移为 `p*C*W + c*W + w`。

```python
from torch_npu._inductor import dvm

B, P, C, H, W = 16, 4, 8, 127, 64

@dvm.kernel(ktype="vector", dyn_shape=False)
def sum_bh(k):
    x = k.view_load([C, P, W, B, H],
                    [H*W, C*H*W, 1, P*C*H*W, W], dvm.float32)
    total = k.sum(x, [3, 4], True)
    k.view_store(total, [W, C*W, 1, 1, 1])

# x 为上述连续 NPU 输入，out 为预分配的连续 FP32 [P,C,W]
# sum_bh.run(x, out)
```

在本机独立 DVM 的 `Ascend950DT_9581` host codegen 上，该形状的普通版本包含清零、`SumY` 和 `store_atomic`。View 版本生成 `view_load_x → ClrPad → SumX → view_store`。验收时在目标 PTA 上用相同输入对照 `torch.sum`，用非对称尺寸、带正负号数据和非对齐尾块查错轴与写回，再比较全部 DEVICE 任务耗时（含清零/归并）。区分 DAS 中的一条原子指令与运行时重复执行的原子次数。

## 适配真实签名

- 示例仅接收指定 shape、stride 和 dtype。对 FP16/BF16 保留原算子的累加精度、输出舍入和 keepdim 契约。
- 本机 PTA 的静态 `TorchKernelPy::ViewLoad` 使用 builder 的声明，动态 `DynKernelPy::ViewLoad` 从实参取 sizes/strides。动态版本传真实的零拷贝 `x.permute(...)` 视图并核对运行时轴；切换 shape 后重新验证。静态 `.run` 对非连续输出可能创建临时 Tensor，优先传入匹配容量的连续输出底层缓冲，避免重复转置写回。
- 查当前版 `torch_npu/csrc/inductor/dvm/pybind_api.cpp` 的 ViewLoad/ViewStore/Run，以及 `kobj.das()`。含 `PlanReduction` 的后端可能已自动重排，仍以普通 builder 作同版基线。
