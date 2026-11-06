# ATen_Size_Profiler v0.2.2

## 一、使用方式

环境要求：Python >= 3.10，PyTorch >= 2.9（Eager 模式）。工具包不声明
PyTorch 依赖，pip 不会安装或替换 PyTorch，请先在环境中安装与 CPU/CUDA/ROCm
匹配的 PyTorch。

1. 安装工具包（PyYAML、openpyxl 会按需自动安装）：

```bash
python -m pip install \
  https://github.com/yaozh1/ATen_Size_Profiler/releases/download/v0.2.2/aten_size_profiler-0.2.2-py3-none-any.whl
```

2. 修改模型运行脚本：

在模型执行之前导入 ATen Size Profiler：

```python
import aten_size_profiler
```

举例：

```python
# train.py
import torch
from torch import nn

import aten_size_profiler  # 在模型执行前导入，导入即开始采集

model = nn.Linear(16, 4)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(10):
    optimizer.zero_grad()
    loss = model(torch.randn(8, 16)).sum()
    loss.backward()
    optimizer.step()

# 原有训练代码不需要改动，进程退出时自动导出报表
```

3. 启动脚本，自动采集：

一般模型需要设置这几个环境变量：

```bash
# 启用 Profiler（默认即为 true）
export ATEN_SIZE_PROFILER_ENABLED=true

# 只采集第 2 次 iteration（从 0 开始计数）
export ATEN_SIZE_PROFILER_TARGET_ITER=1

# 运行脚本，自动完成采集
python train.py
```

训练场景需要增加（推理使用默认的 `forward`）：

```bash
export ATEN_SIZE_PROFILER_ITERATION_MODE=optimizer_step
```

需要记录 4D/5D 输入的内存布局（如视觉类模型的 NCHW/NHWC）时增加：

```bash
export ATEN_SIZE_PROFILER_RECORD_MEMORY_FORMAT=true
```

4. 分布式训练用法：

单机多卡 DDP：

```bash
# 启用 Profiler
export ATEN_SIZE_PROFILER_ENABLED=true

# 采集第 2 次 iteration
export ATEN_SIZE_PROFILER_TARGET_ITER=1

# 使用训练模式
export ATEN_SIZE_PROFILER_ITERATION_MODE=optimizer_step

# 启动 DDP
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 train.py
```

5. 查看采集结果：

```text
当前工作目录/
└── aten_size_reports/
    └── aten_size_profile_size.xlsx
```

当前工作目录指启动 `python` 或 `torchrun` 的目录。分布式运行时每个 Rank
写入独立文件，例如 `aten_size_profile_size_rank0.xlsx`、
`aten_size_profile_size_rank1.xlsx`。

6. 生命周期控制（可选）：

```python
aten_size_profiler.pause_profiler()             # 暂停采集
aten_size_profiler.resume_profiler()            # 恢复采集
path = aten_size_profiler.stop_profiler()       # 停止并导出，返回报表路径
aten_size_profiler.stop_profiler(export=False)  # 停止且不导出
```

未调用 `stop_profiler()` 时，进程退出时自动导出一次。

## 二、环境变量

| 环境变量 | 含义 | 默认值 | 常用值 |
|---|---|---|---|
| `ATEN_SIZE_PROFILER_ENABLED` | 是否启用 Profiler | `true` | `true` / `false` |
| `ATEN_SIZE_PROFILER_TARGET_ITER` | 仅采集指定 iteration（从 0 开始） | `null`（持续采集） | `0`、`1`、`10` |
| `ATEN_SIZE_PROFILER_ITERATION_MODE` | iteration 的划分方式：`forward` 以一次顶层模型 forward 为一个 iteration，不包含 backward 和优化器更新；`optimizer_step` 从 forward 开始到 `optimizer.step()` 结束为一个 iteration | `forward` | `forward` / `optimizer_step` |
| `ATEN_SIZE_PROFILER_OUTPUT_DIR` | Excel 输出目录 | `./aten_size_reports` | `./aten_size_reports` |
| `ATEN_SIZE_PROFILER_OUTPUT_FILENAME` | Excel 文件名 | `aten_size_profile_size.xlsx` | `model_size.xlsx` |
| `ATEN_SIZE_PROFILER_CONFIG` | YAML 配置文件路径 | 未设置（读取当前工作目录下的 `aten_size_profiler.yaml`） | `/path/to/config.yaml` |
| `ATEN_SIZE_PROFILER_RECORD_SCALAR_ARGS` | 是否记录标量参数并参与去重 | `true` | `true` / `false` |
| `ATEN_SIZE_PROFILER_RECORD_STRIDE` | 是否记录 Tensor stride | `true` | `true` / `false` |
| `ATEN_SIZE_PROFILER_RECORD_STACK` | 是否记录 Python 调用栈 | `false` | `true` / `false` |
| `ATEN_SIZE_PROFILER_RECORD_MEMORY_FORMAT` | 是否采集 4D/5D Tensor 内存布局 | `false` | `true` / `false` |

补充说明：

- `TARGET_ITER` 不设置时持续采集，建议正式分析时指定目标 iteration，避免
  内存持续增长。
- `CONFIG` 不设置时，会尝试读取当前工作目录下的 `aten_size_profiler.yaml`，
  文件不存在则使用默认配置；设置了但文件不存在时报 `FileNotFoundError`。
- 环境变量优先级高于 YAML 配置。
- `OUTPUT_DIR` 为相对路径时按当前工作目录解析。`OUTPUT_FILENAME` 必须是
  不含目录、扩展名为 `.xlsx` 的文件名，`ITERATION_MODE` 只能是 `forward`
  或 `optimizer_step`，否则直接报 `ValueError`。
- DDP 多进程运行时，Excel 文件名会自动添加 `_rank<N>` 后缀，防止不同 Rank
  的报表互相覆盖。

也支持 YAML 配置（环境变量按字段覆盖 YAML）：

```yaml
enabled: true
target_iter: 1
iteration_mode: forward
output_dir: ./aten_size_reports
record_stride: true
record_stack: false
record_scalar_args: true
record_memory_format: false
```

## 三、输出解释

报表包含 `summary`、`unique_shapes`、`detail`、`input_tensors` 和
`metadata` 五个 Sheet。`input_memory_layouts`、`memory_layout` 两列只在
开启 `RECORD_MEMORY_FORMAT` 时出现。目标 iteration 未完整执行（例如运行
步数不足或中途异常）时，报表不含调用记录，`metadata.iteration_complete`
为 `false`。

### summary — 算子汇总表

每行代表一种 ATen 算子，统计其在采集窗口内的总体调用情况。

| 表头字段 | 含义 | 数据来源 / 计算方式 |
|---|---|---|
| `op_name` | ATen 算子名称，包含 overload，如 `aten.mm.default` | TorchDispatchMode 拦截的算子 Schema |
| `call_count` | 该算子的总调用次数 | 按 `op_name` 统计 CallRecord 数量 |
| `unique_shape_count` | 该算子的不同输入组合数量 | 按输入 Tensor 的 shape、dtype、device、stride，以及可选的 `scalar_args` 去重 |
| `input_tensor_count_total` | 该算子所有调用涉及的输入 Tensor 总数 | 累加每次调用的 Tensor 输入数量 |
| `input_logical_bytes_total` | 该算子所有调用的输入逻辑字节数总和，单位 Byte | 累加所有输入 Tensor 的 numel × element_size |

### unique_shapes — 输入 Shape 分布统计

每行代表某个 ATen 算子的一种唯一输入元数据组合，用于分析高频 Shape。
去重键为按输入顺序排列的各输入 Tensor 的 shape、dtype、device、stride，
加上 `scalar_args`，不含输入路径。

| 表头字段 | 含义 | 数据来源 / 计算方式 |
|---|---|---|
| `op_name` | ATen 算子名称 | TorchDispatchMode |
| `input_shapes` | 该组合中各输入 Tensor 的 Shape 列表 | `tensor.shape` |
| `input_dtypes` | 各输入 Tensor 的数据类型 | `tensor.dtype` |
| `input_devices` | 各输入 Tensor 所在设备 | `tensor.device` |
| `input_strides` | 各输入 Tensor 的 stride 信息 | `tensor.stride()`，受 `record_stride` 配置控制，关闭时为 `null` |
| `count` | 该输入组合出现的次数 | 对相同输入元数据组合进行计数 |
| `op_ratio` | 该输入组合占当前算子调用次数的比例 | `count` / 当前算子的 `call_count` |
| `overall_ratio` | 该输入组合占全部 ATen 调用次数的比例 | `count` / 全部算子的调用总数 |
| `scalar_args` | 算子调用的标量参数及其位置 | Collector 递归解析 args、kwargs 中的非 Tensor 标量参数，记录参数路径及对应值 |
| `input_memory_layouts` | 各输入 Tensor 的内存布局列表 | 通过 `is_contiguous()` 判断 Tensor 内存布局 |

tips：`input_memory_layouts` 仅对 4D/5D Tensor 判断连续性。`NCHW`/`NCDHW`
为普通连续，`NHWC`/`NDHWC` 为 Channels Last，`ambiguous` 为两者均满足
`is_contiguous()`，`non_contiguous` 为均不满足，`null` 为未分类，`mixed`
为聚合后布局不一致。

### detail — 算子调用明细

按 ATen 算子的调用顺序记录，每次调用占一行，无论是否有 Tensor 输入。

| 表头字段 | 含义 | 数据来源 / 计算方式 |
|---|---|---|
| `call_id` | 算子调用编号，从 0 开始 | Collector 按采集顺序生成 |
| `op_name` | ATen 算子名称 | TorchDispatchMode |
| `input_count` | 本次调用涉及的 Tensor 输入数量 | `len(call.inputs)` |
| `input_paths` | 输入 Tensor 在算子参数中的位置 | 递归解析 args、kwargs，如 `args[0]` |
| `input_shapes` | 本次调用所有输入 Tensor 的 Shape | `tensor.shape` |
| `input_dtypes` | 所有输入 Tensor 的数据类型 | `tensor.dtype` |
| `input_devices` | 所有输入 Tensor 的设备信息 | `tensor.device` |
| `input_strides` | 所有输入 Tensor 的 stride | `tensor.stride()` |
| `input_numels` | 各输入 Tensor 的元素数量 | `tensor.numel()` |
| `input_element_sizes` | 各输入 Tensor 单个元素占用的字节数 | `tensor.element_size()` |
| `input_logical_bytes` | 本次调用所有输入 Tensor 的逻辑字节数总和 | sum(numel × element_size) |
| `stack` | 算子调用时的 Python 调用栈 | `traceback.format_stack(limit=32)`；需启用 `record_stack` |
| `iteration` | 本次调用所属的 iteration 编号 | IterationController 记录，导出时可使用配置值补充 |
| `rank` | 本次调用所属的全局 Rank | 分布式 Rank 识别结果，由导出器写入 |
| `scalar_args` | 当前 ATen 调用实际记录的标量参数 | Collector 递归解析 args、kwargs 中的非 Tensor 标量参数，记录参数路径及对应值 |
| `input_memory_layouts` | 本次 ATen 调用所有输入 Tensor 的内存布局 | 按输入顺序记录，非 4D/5D Tensor 使用 `null` 占位 |

### input_tensors — Tensor 输入明细

按 ATen 算子的调用顺序记录，每个输入 Tensor 占一行；没有 Tensor 输入的
调用不产生记录。

| 表头字段 | 含义 | 数据来源 / 计算方式 |
|---|---|---|
| `call_id` | 所属 ATen 调用的编号 | 对应 `detail.call_id` |
| `op_name` | 所属 ATen 算子名称 | 对应 CallRecord 的算子名称 |
| `input_index` | 本次调用中 Tensor 输入的顺序编号，从 0 开始 | Collector 对展开后的 Tensor 输入依次编号 |
| `input_path` | Tensor 在原始参数中的位置 | 递归遍历 args、kwargs，如 `args[0]`、`args[1][0]` |
| `shape` | Tensor 的维度及各维长度 | `tensor.shape` |
| `dtype` | Tensor 数据类型 | `tensor.dtype` |
| `device` | Tensor 所在设备 | `tensor.device` |
| `stride` | Tensor 各维度的步长 | `tensor.stride()` |
| `numel` | Tensor 元素总数 | `tensor.numel()` |
| `element_size` | 每个元素占用字节数，单位 Byte | `tensor.element_size()` |
| `logical_bytes` | Tensor 的逻辑数据大小，单位 Byte | numel × element_size |
| `iteration` | 当前 Tensor 所属调用的 iteration | 对应 CallRecord 的 iteration |
| `rank` | 当前 Tensor 所属调用的全局 Rank | 分布式 Rank 识别结果 |
| `memory_layout` | 当前 Tensor 的内存布局 | 仅对 4D/5D Tensor 使用 `is_contiguous()` 判断 |

### metadata — 采集环境与配置信息

每行一个字段，表头为 `key` 和 `value`，按 key 排序。

| 字段（key） | 含义 | 数据来源 / 计算方式 |
|---|---|---|
| `call_count` | 本次采集的 ATen 调用总次数 | `len(collector.calls)` |
| `config_file` | 加载的 YAML 配置文件路径 | `ProfilerConfig.config_file` |
| `configured_enabled` | Profiler 配置是否启用 | `ProfilerConfig.enabled` |
| `configured_output_dir` | 配置的 Excel 输出目录 | `ProfilerConfig.output_dir` |
| `configured_output_filename` | 配置的 Excel 文件名 | `ProfilerConfig.output_filename` |
| `created_at` | Excel 报表生成时间，UTC | `datetime.now(timezone.utc).isoformat()` |
| `cuda_available` | PyTorch 是否检测到可用 GPU | `torch.cuda.is_available()` |
| `cuda_version` | PyTorch 对应 CUDA 版本 | `torch.version.cuda` |
| `devices` | 本次采集记录中实际出现的 Tensor 设备集合 | 遍历所有 CallRecord 的输入 Tensor，去重并排序 |
| `distributed_initialized` | Profiler 观测或缓存的分布式 Process Group 初始化状态 | `RankInfo.initialized` |
| `forward_root_module` | 自动识别的 forward 根模块 | `IterationHooks.root_module_name` |
| `global_rank` | 当前进程的全局 Rank | `RankInfo.rank`，来自 `torch.distributed` 或环境变量 |
| `hip_version` | PyTorch 对应 HIP 版本 | `torch.version.hip` |
| `iteration` | 报表对应的 iteration | 导出时传入的 iteration，通常为目标 iteration |
| `iteration_complete` | 目标 iteration 是否完整完成 | 判断 `IterationController.status()["phase"] == "FINISHED"`；未配置状态机时为空 |
| `iteration_mode` | iteration 识别模式 | `ProfilerConfig.iteration_mode` |
| `iteration_phase` | 当前 iteration 状态 | `IterationController.status()["phase"]`；未配置状态机时为 `not configured` |
| `local_rank` | 当前进程的本地 Rank | `RankInfo.local_rank`，通常来自 `LOCAL_RANK` 环境变量 |
| `other_top_level_modules` | 识别到的其他顶层模块 | `sorted(IterationHooks.other_top_level_modules)` |
| `output_file` | Excel 输出路径 | 导出器实际使用的文件路径 |
| `parameterized_non_root_modules` | 包含参数的非根模块 | `sorted(IterationHooks.parameterized_non_root_modules)` |
| `platform` | 操作系统及平台信息 | `platform.platform()` |
| `python_version` | Python 版本 | `sys.version.split()[0]` |
| `pytorch_version` | PyTorch 版本 | `torch.__version__` |
| `rank` | 当前报表对应的全局 Rank | 导出器传入的 `RankInfo.rank` |
| `record_memory_format` | 是否开启 4D/5D Tensor 内存布局采集 | `ProfilerConfig.record_memory_format` |
| `record_scalar_args` | 是否启用标量参数采集 | `ProfilerConfig.record_scalar_args` |
| `record_stack` | 是否启用 Python 调用栈采集 | `ProfilerConfig.record_stack` |
| `record_stride` | 是否启用 stride 信息采集 | `ProfilerConfig.record_stride` |
| `target_iter` | 配置中指定的目标 iteration | `ProfilerConfig.target_iter` |
| `tool_version` | Profiler 工具版本 | `aten_size_profiler.__version__` |
| `truncated_cells` | 超长单元格截断数量 | `_ExportStats.truncated_cells`，每截断一个单元格加 1 |
| `truncated_sheets` | 发生截断的 Sheet | `_ExportStats.truncated_sheets`，记录涉及的 Sheet 并排序 |
| `world_size` | 分布式进程总数 | `RankInfo.world_size`，来自 `torch.distributed` 或环境变量 |

## 四、已知限制

1. 默认持续采集会增长内存占用；Excel 单 Sheet 超过 1,048,576 行时导出失败。

2. 超过 32,767 字符的单元格会截断，并以 `...<TRUNCATED>` 结尾，截断信息
   写入 metadata。

3. `optimizer_step` 模式下，训练 step 之间的验证 forward 可能混入下一个
   iteration。

   `optimizer_step` 模式从 forward 开始到 `optimizer.step()` 结束计为一次
   iteration。训练 step 之间穿插的验证 forward（例如每个 step 后在
   `torch.no_grad()` 下执行 eval）会计入下一个训练 iteration，污染目标
   iteration 的统计。建议在目标 iteration 完成后再进行验证；也可以在验证
   前调用 `pause_profiler()`，并紧贴下一次训练 forward 之前调用
   `resume_profiler()`。该写法已在单 Optimizer CPU 示例中确认，调用记录的
   算子和输入 shape 与无验证运行一致：

   ```python
   for batch, target in dataloader:
       optimizer.zero_grad()
       aten_size_profiler.resume_profiler()
       loss_fn(model(batch), target).backward()
       optimizer.step()

       aten_size_profiler.pause_profiler()
       with torch.no_grad():
           model.eval()
           evaluate(model)
           model.train()
   ```

   `resume_profiler()` 与 forward 之间的调用（例如数据准备）仍会计入目标
   iteration，因此数据准备和 `optimizer.zero_grad()` 应放在
   `resume_profiler()` 之前。未暂停时调用 `resume_profiler()` 没有影响，
   首轮循环和未执行验证的轮次可以使用同一写法。

4. 支持范围：面向 PyTorch Eager、标准 `nn.Module` 推理和标准单 Optimizer
   训练。暂不承诺支持 `torch.compile`、复杂训练引擎、梯度累积、多个
   Optimizer、自定义 iteration 边界、多个并列顶层模型 forward 和跨线程执行。

5. `forward` 模式以首次出现的顶层 `nn.Module` 作为 forward 边界。如果模型
   之前还调用了带参数的独立顶层模块（例如预处理层），该模块会被误认为根
   模块；此时会发出 `RuntimeWarning`，并在 metadata 的
   `forward_root_module` 等字段中记录，建议改用 `optimizer_step` 或把预处理
   放进模型内部。

6. 只记录 `aten::` 命名空间的算子，`c10d` 等通信算子不进入报表。与默认值
   相同的参数通常不会传到 dispatch 层，因此不记录。

7. `logical_bytes` 是 numel × element_size，不代表实际显存占用，同一
   Tensor 多次传入会重复累加。

8. 相同输出路径的多次运行会互相覆盖，需要保留时请更换输出目录或文件名。

9. 采集会给每个 ATen 调用增加 Python 层开销，报表不适合用于计时分析。
