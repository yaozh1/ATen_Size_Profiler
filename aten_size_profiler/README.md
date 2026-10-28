# ATen Size Profiler 使用说明

ATen Size Profiler 用于采集 PyTorch Eager 模式下可观测的 ATen
算子输入信息，并导出 Excel 报表。

安装、运行环境、测试和 GitHub Actions 见项目根目录的 `README.md`；本文件
说明配置、使用方式、输出报表和当前限制。

## 最小使用方式

在模型执行前导入一次：

```python
import aten_size_profiler
```

Profiler 会在导入时自动安装，并在进程退出时导出 Excel 报表。可以使用
YAML 配置：

```bash
cat >/tmp/aten_size_profiler.yaml <<'YAML'
enabled: true
output_dir: ./aten_size_reports
output_filename: model_size.xlsx
record_stride: true
record_stack: false
record_scalar_args: true
iteration_mode: forward
target_iter: 0
YAML

ATEN_SIZE_PROFILER_CONFIG=/tmp/aten_size_profiler.yaml \
python your_model.py
```

环境变量会逐项覆盖 YAML 配置：

```bash
ATEN_SIZE_PROFILER_ENABLED=true \
ATEN_SIZE_PROFILER_OUTPUT_DIR=./aten_size_reports \
ATEN_SIZE_PROFILER_TARGET_ITER=10 \
ATEN_SIZE_PROFILER_ITERATION_MODE=forward \
python your_model.py
```

## 配置

配置文件路径由 `ATEN_SIZE_PROFILER_CONFIG` 指定；未指定时读取当前工作
目录下的 `aten_size_profiler.yaml`（存在时）。每个配置项都可以用
`ATEN_SIZE_PROFILER_<配置项大写>` 环境变量覆盖。

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `enabled` | `true` | 是否采集。 |
| `output_dir` | `aten_size_reports` | 报表目录，相对路径按当前工作目录解析。 |
| `output_filename` | `aten_size_profile_size.xlsx` | 报表文件名，规则见下文。 |
| `record_stride` | `true` | 是否记录 stride。 |
| `record_stack` | `false` | 是否记录 Python 调用栈。 |
| `record_scalar_args` | `true` | 是否记录标量参数并纳入 `unique_shapes` 去重键。 |
| `record_memory_format` | `false` | 是否记录输入 Tensor 的内存布局，规则见“内存布局”。 |
| `target_iter` | `null` | 采集的 iteration，从 0 开始；`null` 表示不启用 iteration 窗口，即持续采集。 |
| `iteration_mode` | `forward` | `forward` 或 `optimizer_step`。 |

- `output_filename` 必须是不含目录的文件名，扩展名为 `.xlsx`（大小写
  不敏感），且扩展名前的文件名不能为空，例如 `model_size.xlsx`。
  `report.csv`、`report`、`.xlsx`、`sub/report.xlsx` 或首尾带空白的名称会
  直接报配置错误。目录请通过 `output_dir` 设置。直接调用
  `export_collector()` 时，目标路径同样必须以 `.xlsx` 结尾。
- 未显式设置 `output_dir` 时，`aten_size_reports/` 按当前工作目录解析；
  如果需要输出到脚本所在目录，请显式指定绝对路径。
- 标量参数包括数值、布尔、字符串、`None`，以及 `dtype`、`device`、
  `memory_format` 和 `layout`。
- `iteration_mode` 只能是 `forward` 或 `optimizer_step`；其他值会直接报
  配置错误。本版本只支持通过 import 和配置自动采集，不提供需要修改模型
  代码的手动迭代接口。

## 推理和训练

标准推理配置示例：

```yaml
enabled: true
iteration_mode: forward
target_iter: 10
output_dir: ./aten_size_reports
record_stride: true
record_stack: false
```

标准单 Optimizer 训练将 `iteration_mode` 设置为 `optimizer_step`。
模型代码仍然只需要 `import aten_size_profiler`。

`forward` 模式按一次运行中首次出现的顶层 `nn.Module` 作为业务模型
forward 边界，独立的 loss 模块不会被当作新的 iteration。如果模型之前还
调用了带参数的独立顶层模块（例如作为 `nn.Module` 的预处理层），该模块会
被误认为根模块。此时 Profiler 会发出一次 `RuntimeWarning`，并在 `metadata`
中记录 `forward_root_module`、`other_top_level_modules` 和
`parameterized_non_root_modules`，便于确认报表是否覆盖了目标模型；
这类场景建议改用 `optimizer_step` 或把预处理放进模型内部。

标准训练建议使用 `optimizer_step`；该模式会把 `GradScaler.step()` 的正常
更新和跳过更新都计为一次训练 iteration。

`optimizer_step` 模式从 forward 开始到 `optimizer.step()` 结束计为一次
iteration。训练 step 之间穿插的验证 forward（例如每个 step 后在
`torch.no_grad()` 下执行 eval）会计入下一个训练 iteration，污染目标
iteration 的统计。建议在目标 iteration 完成后再进行验证；也可以在验证
代码前后调用 `pause_profiler()` 和 `resume_profiler()`，该方式已在单
Optimizer CPU 示例中确认可排除验证调用：

```python
for batch, target in dataloader:
    optimizer.zero_grad()
    loss_fn(model(batch), target).backward()
    optimizer.step()

    aten_size_profiler.pause_profiler()
    with torch.no_grad():
        model.eval()
        evaluate(model)
        model.train()
    aten_size_profiler.resume_profiler()
```

采用这种方式时，`resume_profiler()` 之后到下一次训练 forward 之前的调用
（例如数据准备）也会计入目标 iteration。

## 输出报表

输出报表包含 `summary`、`unique_shapes`、`detail`、`input_tensors` 和
`metadata` 五个 Sheet。分布式运行时，每个 Global Rank 会在
`output_filename` 的基础上写入独立的 `*_rank<N>.xlsx` 文件：

```bash
torchrun --standalone --nproc_per_node=2 your_model.py
```

`metadata` 中的 `iteration_complete` 表示目标 iteration 是否完整完成。
如果 loss、backward 或 optimizer step 阶段异常导致目标 iteration 未完成，
导出报表会清空不完整调用，并将该字段设为 `false`。

`unique_shapes` 和 `summary.unique_shape_count` 使用同一去重键：

- 按输入顺序排列的每个输入 Tensor 的 shape、dtype、device、stride，
  不包含输入路径。输入顺序不同（例如 `mul(a, b)` 和 `mul(b, a)` 且两者
  元数据不同）时分为两行。
- 加上该次调用的 `scalar_args`，即参数位置和值组成的列表，例如
  `[["args[1][0]","1"]]`。与默认值相同的参数通常不会传到 dispatch 层，
  因此 `sum(x, 1)` 与 `sum(x, 1, keepdim=False)` 合并为一行。

配置开关对去重的影响：

| 配置 | 报表中的值 | 去重行为 |
| --- | --- | --- |
| `record_stride: true`（默认） | 实际 stride，例如 `[[6,1]]` | 只有 stride 不同的调用分为多行。 |
| `record_stride: false` | `unique_shapes`、`detail` 中为 `null`，`input_tensors` 中为空 | 只有 stride 不同的调用合并为一行。 |
| `record_scalar_args: true`（默认） | 标量参数列表 | 只有标量参数不同的调用分为多行。 |
| `record_scalar_args: false` | `unique_shapes`、`detail` 中为 `[]` | 只有标量参数不同的调用合并为一行。 |

`get_collector().aggregates` 返回的内存聚合是另一套口径：
`OperatorAggregate.input_signature_counts` 和 `unique_input_count` 按每个
输入的路径、shape、dtype、device 加 `scalar_args` 去重，不含 stride，
因此可能与报表中的 `unique_shape_count` 不同。例如只有 stride 不同的
两次调用，内存聚合计为 1 种，报表计为 2 种。统计结果以 Excel 报表
为准。

同一输出路径的多次运行会互相覆盖，报表不带运行编号；需要保留历史结果时
请更换 `output_dir` 或 `output_filename`。

## 内存布局

`record_memory_format: true`（或
`ATEN_SIZE_PROFILER_RECORD_MEMORY_FORMAT=true`）时，按 PyTorch
`is_contiguous()` 判断每个 4D、5D 输入 Tensor 的内存布局：

| 维度 | `is_contiguous()` / channels-last 判断 | 结果 |
| --- | --- | --- |
| 4D | `True` / `False` | `NCHW` |
| 4D | `False` / `True`（`torch.channels_last`） | `NHWC` |
| 5D | `True` / `False` | `NCDHW` |
| 5D | `False` / `True`（`torch.channels_last_3d`） | `NDHWC` |
| 4D、5D | `True` / `True` | `ambiguous` |
| 4D、5D | `False` / `False` | `non_contiguous` |

`NCHW`/`NHWC` 等名称只表示内存排列顺序，不代表 Tensor 各维度的语义。
`ambiguous` 常见于 C=1、H=W=1 或空 Tensor 等两种格式都成立的情况。
其他维度的 Tensor 不调用 `is_contiguous()`，布局记为 `null`，但仍和原来
一样记录，shape、stride 等列和统计不变。不支持 `is_contiguous()` 的
4D、5D 输入（例如批量稀疏 CSR）同样记为 `null`。

开启后在原有列之后追加一列，五个 Sheet 和其他列不变；关闭时报表列与
0.2.1 相同：

| Sheet | 新列 | 内容 |
| --- | --- | --- |
| `input_tensors` | `memory_layout` | 该输入的布局；非 4D/5D 输入为空。 |
| `detail` | `input_memory_layouts` | 按输入顺序排列的布局列表，非 4D/5D 输入用 `null` 占位，例如 `["NHWC","NHWC",null]`。 |
| `unique_shapes` | `input_memory_layouts` | 该分组内所有调用的布局；同一输入位置布局不一致时为 `mixed`。 |

内存布局不参与 `unique_shapes` 去重，也不改变 `record_stride` 的含义。
`record_stride: true` 时 stride 已在去重键中，同一分组内布局一致；
`record_stride: false` 时 NCHW 和 NHWC 等不同布局的调用可能合并为一行，
此时对应输入显示为 `mixed`，各次调用的实际布局见 `detail` 和
`input_tensors`。`metadata` 的 `record_memory_format` 记录该开关。

判断只读取 Tensor 已缓存的元数据，不访问数据，不触发 GPU 同步。

## 生命周期控制

可以使用 `pause_profiler()` 和 `resume_profiler()` 临时暂停和恢复采集。
`stop_profiler()` 会停止采集并导出一次报表。
如果不需要导出报表，使用 `stop_profiler(export=False)`，包括进程退出
时的兜底处理。

## 当前限制

当前实现面向 PyTorch Eager、标准 `nn.Module` 推理和标准单 Optimizer
训练。暂不承诺支持 `torch.compile`、复杂训练引擎、梯度累积、多个
Optimizer、自定义 iteration 边界、多个并列顶层模型 forward、跨线程执行、
跨节点分布式恢复或 Excel Sheet 自动拆分。

`target_iter` 为 `null` 时持续采集整个进程的全部调用，调用记录保存在
内存中，内存占用随运行时间增长。当任一 Sheet 可能超过 Excel 的
1,048,576 行限制时，Excel 导出会明确失败，不生成报表。长时间运行或
训练场景建议配置 `target_iter`。

单个字符串单元格超过 32,767 字符时会截断为 32,767 字符，并以
`...<TRUNCATED>` 结尾，`metadata` 中的 `truncated_cells` 和
`truncated_sheets` 记录截断数量和所在 Sheet。被截断的 JSON 列无法完整
解析，使用前请检查 `truncated_cells`。

`logical_bytes` 是 `numel * element_size`，不代表实际 GPU 物理显存占用，
同一 Tensor 多次传入会重复累加。采集会给每个被观测的 ATen 调用增加
Python 层开销；当前版本已做算子元数据缓存优化，具体性能收益待在稳定
负载环境下复测。
