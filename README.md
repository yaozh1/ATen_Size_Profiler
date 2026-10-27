# ATen Size Profiler

ATen Size Profiler 采集 PyTorch Eager 模式下 ATen 算子的输入 Tensor
元数据，按算子统计输入 Size 分布并导出 Excel 报表，用于确定需要优先
优化的算子和输入形状。

当前版本：0.2.1。变更见 [`CHANGELOG.md`](CHANGELOG.md)，详细配置和
使用说明见 [`aten_size_profiler/README.md`](aten_size_profiler/README.md)。

## 核心功能

- 基于 TorchDispatchMode 拦截分解后到达 Python dispatch 的 `aten::`
  算子，模型代码只需 `import aten_size_profiler`。
- 每个输入 Tensor 记录输入路径（如 `args[0]`）、shape、dtype、device、
  stride、numel、element_size 和 logical_bytes（`numel * element_size`）；
  每次调用可选记录标量参数和 Python 调用栈；可选记录内存布局（如
  NCHW/NHWC）。
- 支持在推理（`forward`）或标准单 Optimizer 训练（`optimizer_step`）中
  只采集指定的一个 iteration。
- 导出五个 Sheet 的 Excel 报表；分布式运行时每个 Global Rank 独立导出。

## 环境要求

- Python >= 3.10
- PyTorch >= 2.13（Eager 模式）
- PyYAML >= 6.0
- openpyxl >= 3.1

工具不修改 PyTorch，直接使用当前环境中已安装的版本：

- CPU：安装 CPU 版 PyTorch 后直接使用。
- NVIDIA GPU：安装与 CUDA 匹配的 PyTorch（尚未验证，见“验证状态”）。
- AMD GPU：安装与 ROCm 匹配的 PyTorch。PyTorch 仍使用 `torch.cuda` 设备
  接口，分布式后端使用 `nccl`，底层对应 RCCL。

## 安装

工具是纯 Python 包，可以不安装、直接使用源码，也可以用 pip 安装。两种
方式下模型代码相同，都只需 `import aten_size_profiler`。

| 方式 | 适用场景 |
| --- | --- |
| 直接使用源码 | 定制 PyTorch/ROCm 环境或容器内不便安装包、临时分析；修改源码后立即生效；每个运行环境都要设置 `PYTHONPATH`。 |
| pip 安装 | 长期使用、多个项目共享；不依赖源码目录和 `PYTHONPATH`；更新源码后需重新安装。 |

### 直接使用源码（无需安装）

当前环境需已有 PyTorch、PyYAML 和 openpyxl。把 `PYTHONPATH` 指向项目
根目录，再用环境变量配置采集：

```bash
# 保留环境中已有的 PYTHONPATH
export PYTHONPATH=/path/to/ATen_Size_Profiler${PYTHONPATH:+:$PYTHONPATH}

ATEN_SIZE_PROFILER_ENABLED=true \
ATEN_SIZE_PROFILER_ITERATION_MODE=forward \
ATEN_SIZE_PROFILER_TARGET_ITER=10 \
python your_model.py
```

```python
import aten_size_profiler  # 在模型执行前导入一次
```

上例只采集第 11 次（`target_iter` 从 0 开始）顶层模型 forward，进程退出时
报表写入当前工作目录下的 `aten_size_reports/aten_size_profile_size.xlsx`。
`ENABLED` 默认即为 `true`；`TARGET_ITER` 不设置时持续采集整个进程。训练
场景把 `ITERATION_MODE` 改为 `optimizer_step`。全部配置项见“配置”。

### pip 安装

在项目根目录执行：

```bash
python -m pip install --no-deps .
```

`--no-deps` 避免 pip 替换定制的 PyTorch/CUDA/ROCm 环境。普通环境可以
去掉该参数，由 pip 按 `pyproject.toml` 安装依赖。

### 检查环境

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
ATEN_SIZE_PROFILER_ENABLED=false \
python -c "import aten_size_profiler as p; print(p.__version__, p.__file__)"
```

第二条确认实际导入的是哪份代码；两种方式同时存在时，`PYTHONPATH` 中的
源码优先。导入即会启用采集，这里设置 `ENABLED=false`，避免进程退出时
生成报表。

## 配置

配置优先级：环境变量 > YAML 文件 > 默认值。YAML 路径由
`ATEN_SIZE_PROFILER_CONFIG` 指定；未指定时读取当前工作目录下的
`aten_size_profiler.yaml`（存在时）。每个配置项都可以用
`ATEN_SIZE_PROFILER_<配置项大写>` 覆盖，例如
`ATEN_SIZE_PROFILER_TARGET_ITER=10`。

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `enabled` | `true` | 是否采集。 |
| `output_dir` | `aten_size_reports` | 报表目录，相对路径按当前工作目录解析。 |
| `output_filename` | `aten_size_profile_size.xlsx` | 不含目录的 `.xlsx` 文件名。 |
| `record_stride` | `true` | 是否记录 stride。 |
| `record_stack` | `false` | 是否记录 Python 调用栈。 |
| `record_scalar_args` | `true` | 是否记录标量参数。 |
| `record_memory_format` | `false` | 是否记录输入 Tensor 的内存布局（`ATEN_SIZE_PROFILER_RECORD_MEMORY_FORMAT`）。 |
| `target_iter` | `null` | 采集的 iteration，从 0 开始；`null` 为持续采集。 |
| `iteration_mode` | `forward` | `forward` 或 `optimizer_step`。 |

非法的 `iteration_mode` 或 `output_filename` 会直接报 `ValueError`。
各配置项的详细规则见包内 README。

## 快速使用

以下示例假设已按“安装”设置 `PYTHONPATH` 或已用 pip 安装。

### 推理

```bash
cat >aten_size_profiler.yaml <<'YAML'
output_filename: model_size.xlsx
iteration_mode: forward
target_iter: 10
YAML

python your_inference.py
```

```python
import aten_size_profiler  # 在模型执行前导入一次

# 原有推理代码保持不变
with torch.no_grad():
    for batch in dataloader:
        output = model(batch)
```

`forward` 模式以首次出现的顶层 `nn.Module` 调用作为 iteration 边界，
上例只采集第 11 次（`target_iter: 10`）模型 forward。进程退出时报表写入
`./aten_size_reports/model_size.xlsx`。

### 训练

标准单 Optimizer 训练使用 `optimizer_step`，从 forward 开始到
`optimizer.step()` 结束计为一次 iteration：

```bash
ATEN_SIZE_PROFILER_ITERATION_MODE=optimizer_step \
ATEN_SIZE_PROFILER_TARGET_ITER=5 \
ATEN_SIZE_PROFILER_OUTPUT_FILENAME=training_size.xlsx \
python your_training.py
```

```python
import aten_size_profiler

for batch, target in dataloader:
    optimizer.zero_grad()
    loss = loss_fn(model(batch), target)
    loss.backward()
    optimizer.step()
```

目标 iteration 未完整执行（例如运行步数不足或中途异常）时，报表不含
调用记录，`metadata.iteration_complete` 为 `false`。

### 单卡与多卡 DDP

单进程运行时报表名即 `output_filename`。分布式运行时每个 Global Rank
写入独立文件，Rank 由 `torch.distributed` 或 `RANK`/`WORLD_SIZE` 环境
变量确定：

```bash
# 沿用上例训练配置
torchrun --standalone --nproc_per_node=2 your_training.py
# 生成 training_size_rank0.xlsx 和 training_size_rank1.xlsx
```

工具不引入额外通信。只记录 `aten::` 命名空间的算子，`c10d` 等其他命名
空间的通信算子不进入报表。

### 生命周期控制

```python
aten_size_profiler.pause_profiler()           # 暂停采集
aten_size_profiler.resume_profiler()          # 恢复采集
path = aten_size_profiler.stop_profiler()     # 停止并导出，返回报表路径
aten_size_profiler.stop_profiler(export=False)  # 停止且不导出
```

未调用 `stop_profiler()` 时，进程退出时自动导出一次。

## Excel 报表

| Sheet | 内容 |
| --- | --- |
| `summary` | 每个算子一行：调用次数、unique shape 组合数、输入 Tensor 总数、logical_bytes 总和。 |
| `unique_shapes` | 每个算子的输入组合及出现次数、算子内占比和全局占比。 |
| `detail` | 每次调用一行：输入路径、shape、dtype、device、stride、numel、logical_bytes、标量参数、调用栈、iteration 和 rank。 |
| `input_tensors` | 每个输入 Tensor 一行，字段同上。 |
| `metadata` | 工具版本、PyTorch/CUDA/HIP 版本、设备、生效配置、iteration 状态、Rank 信息和截断统计。 |

`unique_shapes` 和 `summary.unique_shape_count` 的去重键是：按输入顺序
排列的每个输入 Tensor 的 shape、dtype、device、stride，加上该次调用的
`scalar_args`；不含输入路径。`record_stride: false` 时 stride 记为
`null`，只有 stride 不同的调用合并为一行；`record_scalar_args: false` 时
`scalar_args` 为 `[]`，只有标量参数不同的调用合并为一行。完整规则见包内
README。`logical_bytes` 不代表实际显存占用，同一 Tensor 多次传入会重复
累加。

`record_memory_format: true` 时，按 `is_contiguous()` 和
channels-last 判断把输入分为 `NCHW`/`NHWC`（4D）、`NCDHW`/`NDHWC`（5D）、
`ambiguous`、`non_contiguous`，其他维度分为 `contiguous`/`non_contiguous`；
名称只表示内存排列顺序，不代表维度语义。`input_tensors` 末尾追加
`memory_layout`，`detail` 和 `unique_shapes` 末尾追加
`input_memory_layouts`；关闭时列结构与 0.2.1 相同。布局不参与去重，
`record_stride: false` 时同一分组内布局不一致的输入显示为 `mixed`。详见
包内 README 的“内存布局”。

## 已知限制

- **默认持续采集**：`target_iter` 为 `null` 时记录整个进程的全部调用，
  内存随运行时间增长；任一 Sheet 超过 Excel 的 1,048,576 行时导出直接
  报错，不生成报表。正式采集建议配置 `target_iter`。
- **训练中穿插验证**：`optimizer_step` 模式下，step 之间的验证 forward
  （如 `torch.no_grad()` 下的 eval）会计入下一个训练 iteration。建议在
  目标 iteration 完成后再进行验证；也可以用 `pause_profiler()` 和
  `resume_profiler()` 包住验证代码，该方式已在单 Optimizer CPU 示例中
  确认可排除验证调用，但恢复后到下一次 forward 之前的调用（如数据准备）
  仍会计入，示例见包内 README。
- **超长单元格**：超过 32,767 字符的单元格截断为 32,767 字符并以
  `...<TRUNCATED>` 结尾，`metadata` 的 `truncated_cells` 和
  `truncated_sheets` 记录截断数量和所在 Sheet。
- **去重口径**：`get_collector().aggregates` 中的
  `input_signature_counts` 和 `unique_input_count` 按输入路径、shape、
  dtype、device 和标量参数去重，不含 stride，与 `unique_shapes` 口径
  不同（例如只有 stride 不同的两次调用，内存聚合计为 1 种，报表计为
  2 种）。统计结果以 Excel 报表为准。
- **报表覆盖**：相同输出路径的多次运行会互相覆盖，需要保留时请更换
  `output_dir` 或 `output_filename`。
- **支持范围**：面向 PyTorch Eager、标准 `nn.Module` 推理和标准单
  Optimizer 训练。暂不承诺支持 `torch.compile`、复杂训练引擎、梯度累积、
  多个 Optimizer、自定义 iteration 边界、多个并列顶层模型 forward、跨线程
  执行和跨节点分布式恢复。与默认值相同的参数通常不会传到 dispatch 层，
  因此不记录；Nested Tensor 输入不记录元数据。
- **性能开销**：每个被观测的 ATen 调用都会增加 Python 层开销，GPU 小
  算子场景尤为明显，报表不适合用于计时分析。

## 验证状态

| 范围 | 状态 |
| --- | --- |
| PyTorch 2.13.0（ROCm/HIP 6.3），CPU 与 AMD GPU | `full --distributed` 回归（含稳定性、真实模型推理和训练 CPU/GPU 各 4 组、CPU 双 Rank gloo）、`smoke --installed` 安装包检查、双 Rank RCCL DDP 通过。 |
| PyTorch 2.14.0（HIP 7.2.26333），CPU 与 GPU | 维护者在独立环境中验证：`smoke`、`core` 回归，Mode 生命周期检查，CNN/Transformer 在 CPU/GPU 上的推理和训练，双卡 RCCL DDP，单机 8 卡真实模型 DDP 通过。稳定性检查（`check_stability`）和 `--installed` 不在本次验证范围内。 |
| NVIDIA CUDA、跨节点分布式 | 尚未验证。 |
| GitHub Actions | `release/0.2.1`（`542dea8`）上 `smoke`、`core`、`installed-package` 通过；`full` 仅手动触发，尚未在 GitHub 上运行。 |

## 测试

统一回归入口为 `tests/run_regression.py`，在项目根目录执行：

```bash
python tests/run_regression.py --suite smoke   # 基础采集和边界
python tests/run_regression.py --suite core    # smoke + 导出、iteration、生命周期等
python tests/run_regression.py --suite full    # core + 真实模型和稳定性，耗时最长
python tests/run_regression.py --suite core --distributed  # 追加 CPU 双 Rank torchrun
python tests/run_regression.py --suite smoke --installed   # 追加安装包检查
```

`--installed` 会从当前源码构建 Wheel，安装到临时的
`venv --system-site-packages` 中，在源码目录外运行检查。GPU 和 RCCL
多卡检查使用 `examples/check_distributed_gpu.py` 单独执行。

## GitHub Actions

`.github/workflows/ci.yml` 复用回归入口，在 `ubuntu-latest`、Python 3.10
和 CPU 版 PyTorch 2.13.0 上运行，只做源码验证，不构建发布产物：

| Job | 触发条件 | 内容 |
| --- | --- | --- |
| `smoke` | 所有分支推送、Pull Request、手动触发 | `compileall` + `--suite smoke` |
| `core` | Pull Request；推送到 `main` 或 `release/*` | `--suite core --distributed` |
| `installed-package` | Pull Request；推送到 `main` 或 `release/*` | `--suite smoke --installed` |
| `full` | 仅手动触发（Actions 页面 Run workflow） | `--suite full --distributed` |

GPU 和 RCCL 多卡检查不在 CI 中运行。

## 项目结构

- `aten_size_profiler/`：Python 包源码和详细使用说明。
- `configs/`：YAML 配置示例。
- `examples/`：回归检查脚本（`check_*.py`）。
- `tests/run_regression.py`：统一回归入口。
- `.github/workflows/ci.yml`：GitHub Actions 工作流。

## 许可证

项目尚未确定是否对外开源，当前不包含许可证文件。对外发布前需由项目
维护者确认授权方式。
