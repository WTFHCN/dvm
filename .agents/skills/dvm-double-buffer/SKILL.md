---
name: dvm-double-buffer
description: 为 PyTorch-NPU 的 DVM 自定义算子评估和实现跨 tile 双缓冲。适用于 Ascend950 上 GM→UB 搬运与 Vector 计算串行，需要在 DVM codegen 将循环体展开成两份字节码的场景。
---

# Double buffer：为相邻 tile 生成两份交错的字节码

先生成计算一次的标准 builder，确认完整调用中 MTE2 和 Vector 都有可观的工作、每核连续处理多个 tile，且上一 tile 的计算不依赖下一 tile。检查目标版本是否已对该图生成跨 tile 双缓冲。先查所用 PyTorch-NPU 版本是否暴露跨 tile 双缓冲接口；生成的 DAS 中应能找到交替使用的 UB 地址和对应事件。只生成 Python 算子时保留普通 builder，用 DAS 标出候选循环；允许修改后端时，再在 DVM codegen 落实下列展开。

## 两份字节码各自处理什么

| 配对循环中的指令 | 地址与依赖 |
| --- | --- |
| Load tile 2j → UB0；Load tile 2j+1 → UB1 | 两个 GM 起点、两个不重叠的 UB 区；各自等待该槽可复用，异步发起并记录各自的搬完事件。第二次搬运可与偶数 tile 的计算重叠。 |
| Compute(UB0) → Store 结果 0 | 只等待槽 0 就绪。最后一次读取 UB0 后释放输入槽；输出和 scratch 等到对应消费者完成后再复用。 |
| Compute(UB1) → Store 结果 1 | 使用槽 1 自己的事件和写回偏移。输出无重叠时各写各的；partial reduce 保留合法的累加与最终归并。 |

- 把原来一步一 tile 的迭代改为一步一对，重算 GM 步长、tile 数、每核分工和尾块。两份 Load/Store 都要参与地址重定位。每个逻辑元素仍计算和写回一次；直接复制整个 builder 会重复处理同一输入。
- 配对前计算双份输入、同时存活的输出/scratch、累加器以及展开后字节码预留；核对事件号和指令字段容量。只有一 tile、奇数尾、资源不足或确定性契约有变化时，可保留单缓冲路径；实现尾部单独处理时，最后一次预取也必须在范围内。
- 用不同值标记奇偶 tile，覆盖 1、2、3 个 tile、最后一块不满、两次运行更换地址。检查 DAS 的两个 GM 偏移、两个 UB 地址和独立的 wait/set。随后实测两版总时长及 MTE2/Vector 重叠；若分块缩小、原子冲突增多或受纯带宽限制，保留更快的版本。
