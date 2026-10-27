# 变更记录

## 0.2.1

v0.2.0 发布后的修复版本，不新增采集能力。

### 修复

- `optimizer_step` 模式改用 `Optimizer.register_step_post_hook` 推进
  iteration，修复 LR Scheduler 包装 `step()` 后的 `__func__` 错误和
  Optimizer 对象泄漏。
- `optimizer.step()` 抛异常后重试时，不再混入失败尝试记录的算子；
  Optimizer 或 LR Scheduler 先于 Profiler 创建、Profiler 重复启停、
  新 Optimizer 复用已释放 Optimizer 的对象 id 时同样生效。
- `GradScaler.step()` 的正常更新和跳过更新都计为一次训练 iteration。
- 目标 iteration 未完整完成时，导出前清空不完整调用，`metadata` 记录
  `iteration_phase` 和 `iteration_complete`。
- `forward` 模式以首次出现的顶层 `nn.Module` 作为 forward 边界；出现带
  参数的非根顶层模块时发出 `RuntimeWarning`，`metadata` 记录根模块和其他
  顶层模块。
- `enabled=false` 时不进入 TorchDispatchMode。
- Nested Tensor 等不支持元数据访问的输入不再导致采集崩溃。
- `out=` 参数按算子 schema 的 `is_out` 排除，覆盖 `topk.values`、
  `max.dim_max` 等非 `out` 命名的输出参数。
- `mp.spawn` 等未设置 `RANK` 的场景缓存 Rank 信息，Process Group 销毁后
  导出或期间没有 ATen 算子时仍使用正确的 Rank 文件名。
- `metadata.distributed_initialized` 表示采集或停止期间曾观察到
  Process Group 已初始化，显式停止和退出时自动导出语义一致。
- `stop_profiler()` 只把 Profiler 自身的 Mode 从栈中取出，不再调用用户
  TorchDispatchMode 的 `__exit__`/`__enter__`；无法取出时停止采集并发出
  `RuntimeWarning`。
- Profiler Mode 声明支持 Higher Order Operator，并作为 infra Mode 运行。
- 超过 Excel 32,767 字符的单元格截断为 32,767 字符并以
  `...<TRUNCATED>` 结尾，`metadata` 记录 `truncated_cells` 和
  `truncated_sheets`。
- 导出失败后再次调用 `stop_profiler()` 或退出时自动导出不再抛
  `UnboundLocalError`。
- 缓存每个 ATen 算子的名称、`out=` 参数排除集合和分布式 Rank 信息，减少
  每次调用的 Python 开销；性能收益尚未在稳定负载环境下复测。

### 配置与报表变化

- 新增 `record_scalar_args`（默认 `true`）：记录标量参数，包括数值、布尔、
  字符串、`None`、`dtype`、`device`、`memory_format` 和 `layout`。
- `detail` Sheet 新增 `scalar_args` 列；`unique_shapes` 去重键加入
  `scalar_args`，同一 shape 组合可能因标量参数不同而拆分为多行。
- `metadata` 新增 `record_scalar_args`、`iteration_phase`、
  `iteration_complete`、`forward_root_module`、`other_top_level_modules`、
  `parameterized_non_root_modules`、`truncated_cells` 和
  `truncated_sheets`。

### 兼容性说明

- `iteration_mode` 只接受 `forward` 和 `optimizer_step`，其他值（包括
  `manual`）直接报 `ValueError`。
- `output_filename` 必须是不含目录、扩展名为 `.xlsx` 的文件名；
  `export_collector()` 的目标路径也必须以 `.xlsx` 结尾。不符合时报
  `ValueError`，不会创建目录或文件。
- 超长单元格从写入原值改为截断，下游解析 `detail.input_paths`、`stack`
  等 JSON 列时需检查 `metadata.truncated_cells`。

### 文档、测试和 CI

- 新增根目录 `README.md` 和本文件；`pyproject.toml` 增加
  `readme = "README.md"`。根目录 README 说明直接使用源码（设置
  `PYTHONPATH`）和 pip 安装两种方式；包内 `aten_size_profiler/README.md`
  保留详细配置和使用说明。
- 补充已知限制：`optimizer_step` 模式下训练 step 之间的验证 forward、默认
  持续采集的内存和行数风险、`unique_shapes` 与内存聚合 API 的去重差异、
  多次运行覆盖报表。
- 新增 `tests/run_regression.py` 统一回归入口（`smoke`、`core`、`full`，
  可选 `--distributed`、`--installed`）和对应的 `examples/check_*.py`。
- 新增 GitHub Actions 工作流 `.github/workflows/ci.yml`，复用
  `tests/run_regression.py` 运行 `smoke`、`core`、`installed-package` 和
  手动触发的 `full`，只做源码验证，不包含发布流程。
- `.gitignore` 增加构建和测试产物。

### 验证状态

- PyTorch 2.13.0（ROCm/HIP 6.3）：CPU 和 AMD GPU 上 `full --distributed`
  回归（含稳定性和真实模型推理、训练）、`smoke --installed` 安装包检查、
  双 Rank GPU DDP 检查通过。
- GitHub Actions：`smoke`、`core`、`installed-package` 通过；`full` 仅手动
  触发，尚未在 GitHub 上运行。
- 未验证：PyTorch 2.14、NVIDIA CUDA、8 卡及跨节点分布式。

## 0.2.0

首个发布版本。

- 通过 TorchDispatchMode 采集 `aten::` 算子输入 Tensor 的 shape、dtype、
  device、stride、numel、element_size 和 logical_bytes。
- 导出 `summary`、`unique_shapes`、`detail`、`input_tensors` 和
  `metadata` 五个 Sheet 的 Excel 报表；任一 Sheet 超过 1,048,576 行时报错。
- 支持 YAML 和环境变量配置，`import aten_size_profiler` 时自动启用，进程
  退出时自动导出。
- 支持 `forward` 和 `optimizer_step` 两种模式下的指定 iteration 采集。
- 分布式运行时每个 Global Rank 写入独立的 `*_rank<N>.xlsx` 报表。
- 提供 `pause_profiler()`、`resume_profiler()` 和 `stop_profiler()`。

## 维护说明

- 版本号以 `aten_size_profiler.__version__` 为准，`pyproject.toml` 通过
  动态版本读取，二者无需分别修改。
- 本文件记录源码维护范围，不代表所有硬件、训练框架或分布式场景均已
  支持。
