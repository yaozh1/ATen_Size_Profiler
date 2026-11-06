# ATen Size Profiler

ATen Size Profiler 采集 PyTorch Eager 模式下 ATen 算子的输入 Tensor
元数据（shape、dtype、device、stride 等），按算子统计输入 Size 分布并导出
Excel 报表，用于确定需要优先优化的算子和输入形状。模型代码只需
`import aten_size_profiler`。

当前版本：0.2.2（tag `v0.2.2`）。相比 0.2.1：

- 新增 `record_memory_format`，可记录 4D/5D 输入的内存布局（NCHW/NHWC 等）。
- PyTorch 最低版本由 2.13 降为 2.9。
- 修复设置 `target_iter` 时与 `torch.profiler.profile(with_stack=True)`
  同时使用导致进程 SIGSEGV 退出的问题。
- 工具包不再声明 PyTorch 依赖，安装时不会安装或替换环境中的 PyTorch。

变更见 [`CHANGELOG.md`](CHANGELOG.md)；使用方式、环境变量和报表字段见
[`aten_size_profiler/README.md`](aten_size_profiler/README.md)。

## 环境要求

- Python >= 3.10
- PyTorch >= 2.9（Eager 模式），需预先安装与 CPU/CUDA/ROCm 匹配的版本
- PyYAML >= 6.0、openpyxl >= 3.1（pip 安装时自动安装）

## 安装

安装 Release 中的 wheel：

```bash
python -m pip install \
  https://github.com/yaozh1/ATen_Size_Profiler/releases/download/v0.2.2/aten_size_profiler-0.2.2-py3-none-any.whl
```

也可以在项目根目录执行 `python -m pip install .` 从源码安装，或者不安装，
直接把 `PYTHONPATH` 指向项目根目录：

```bash
export PYTHONPATH=/path/to/ATen_Size_Profiler${PYTHONPATH:+:$PYTHONPATH}
```

确认实际导入的版本和位置（导入即启用采集，这里关闭以免退出时生成报表）：

```bash
ATEN_SIZE_PROFILER_ENABLED=false \
python -c "import aten_size_profiler as p; print(p.__version__, p.__file__)"
```

## 快速使用

在模型执行前导入一次：

```python
import aten_size_profiler
```

用环境变量指定要采集的 iteration，然后正常启动：

```bash
# 只采集第 2 次 iteration（从 0 开始计数）
export ATEN_SIZE_PROFILER_TARGET_ITER=1
# 训练使用 optimizer_step；推理使用默认的 forward
export ATEN_SIZE_PROFILER_ITERATION_MODE=optimizer_step
python train.py
```

进程退出时，报表写入当前工作目录下的
`aten_size_reports/aten_size_profile_size.xlsx`；分布式运行时每个 Rank
写入独立的 `*_rank<N>.xlsx`。也支持 YAML 配置，环境变量优先于 YAML。

## Excel 报表

| Sheet | 内容 |
| --- | --- |
| `summary` | 每个算子一行：调用次数、不同输入组合数、输入 Tensor 总数、logical_bytes 总和。 |
| `unique_shapes` | 每个算子的输入组合及出现次数、算子内占比和全局占比。 |
| `detail` | 每次调用一行：输入路径、shape、dtype、device、stride、numel、logical_bytes、标量参数、调用栈、iteration 和 rank。 |
| `input_tensors` | 每个输入 Tensor 一行。 |
| `metadata` | 工具版本、PyTorch/CUDA/HIP 版本、生效配置、iteration 状态、Rank 信息和截断统计。 |

## 已知限制

- 面向 PyTorch Eager、标准 `nn.Module` 推理和标准单 Optimizer 训练；暂不
  承诺支持 `torch.compile`、复杂训练引擎、梯度累积、多个 Optimizer 和自定义
  iteration 边界。
- 不设置 `target_iter` 时持续采集，内存随运行时间增长；任一 Sheet 超过
  Excel 的 1,048,576 行时导出失败。
- `optimizer_step` 模式下，step 之间的验证 forward 会计入下一个
  iteration，可用 `pause_profiler()`/`resume_profiler()` 排除。
- 只记录 `aten::` 算子；采集有 Python 层开销，报表不适合用于计时分析。

完整说明见包内 README。

## 验证状态

| 范围 | 状态 |
| --- | --- |
| 0.2.2 修复 | PyTorch 2.13.0 CPU 上 `core --distributed` 回归通过（含新增的 torch profiler 检查）。 |
| PyTorch 2.13.0（ROCm/HIP 6.3） | 0.2.1：CPU 与 AMD GPU 上 `full --distributed` 回归、安装包检查、双 Rank RCCL DDP 通过。 |
| PyTorch 2.14.0（HIP 7.2） | 0.2.1：维护者验证 `smoke`、`core` 回归，CPU/GPU 推理和训练，双卡及单机 8 卡 DDP 通过。 |
| PyTorch 2.9.0 及以上 | 维护者确认支持。 |
| NVIDIA CUDA、跨节点分布式 | 尚未验证。 |

## 测试

统一回归入口为 `tests/run_regression.py`，在项目根目录执行：

```bash
python tests/run_regression.py --suite core --distributed
```

`--suite` 可选 `smoke`、`core`、`full`；`--installed` 会构建 wheel 并在
源码目录外检查。GitHub Actions（`.github/workflows/ci.yml`）在 CPU 版
PyTorch 2.13.0 上运行 `smoke`、`core` 和安装包检查，`full` 仅手动触发；
GPU 和多卡检查不在 CI 中运行。

## 项目结构

- `aten_size_profiler/`：Python 包源码和使用说明。
- `configs/`：YAML 配置模板，不会被自动读取。
- `examples/`：回归检查脚本（`check_*.py`）。
- `tests/run_regression.py`：统一回归入口。

## 许可证

项目尚未确定是否对外开源，当前不包含许可证文件。对外发布前需由项目
维护者确认授权方式。
