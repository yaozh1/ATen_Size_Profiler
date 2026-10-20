# ATen Size Profiler

ATen Size Profiler 用于采集 PyTorch Eager 模式下可观测的 ATen
算子输入信息，并导出 Excel 报表。

## 项目结构

项目根目录包含构建配置、示例和配置文件；`aten_size_profiler` 目录包含
Profiler 的 Python 包源码和本说明文件。

构建 Wheel 时请在项目根目录执行命令。

## 安装

在已经提供所需 PyTorch 版本的环境中构建并安装 Wheel：

```bash
python -m pip wheel --no-deps --no-build-isolation . -w dist
python -m pip install --no-deps dist/aten_size_profiler-0.2.0-py3-none-any.whl
```

定制版 PyTorch/HIP 环境建议使用 `--no-deps`，避免 pip 替换现有 PyTorch。
普通环境可以去掉该参数，由 pip 根据声明安装 `torch`、`PyYAML` 和
`openpyxl`。

### 运行环境

本工具依赖已经安装好的 PyTorch 环境，不要求执行项目内的额外环境脚本。

- CPU：安装 CPU 版 PyTorch 后直接使用。
- NVIDIA GPU：安装与 CUDA 匹配的 PyTorch 后直接使用。
- AMD GPU：安装与 ROCm 匹配的 PyTorch 后直接使用；PyTorch 仍使用
  `torch.cuda` 设备接口，分布式后端使用 `nccl`，底层对应 RCCL。

可以使用下面的命令检查当前 PyTorch 环境：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 最小使用方式

在模型执行前导入一次：

```python
import aten_size_profiler
```

Profiler 会自动安装，并在进程退出时导出 Excel 报表。可以使用 YAML
配置：

```bash
cat >/tmp/aten_size_profiler.yaml <<'YAML'
enabled: true
output_dir: ./aten_size_reports
output_filename: model_size.xlsx
record_stride: true
record_stack: false
iteration_mode: forward
target_iter: 0
YAML

ATEN_SIZE_PROFILER_CONFIG=/tmp/aten_size_profiler.yaml \
python your_model.py
```

安装 Wheel 后，上述用法不依赖项目源码目录，也不需要设置
`PYTHONPATH=.`。

环境变量会逐项覆盖 YAML 配置：

```bash
ATEN_SIZE_PROFILER_ENABLED=true \
ATEN_SIZE_PROFILER_OUTPUT_DIR=./aten_size_reports \
ATEN_SIZE_PROFILER_TARGET_ITER=10 \
ATEN_SIZE_PROFILER_ITERATION_MODE=forward \
python your_model.py
```

支持的配置项包括：
`enabled`、`output_dir`、`output_filename`、`record_stride`、
`record_stack`、`target_iter` 和 `iteration_mode`。
`target_iter` 从 0 开始。默认配置文件名为当前工作目录下的
`aten_size_profiler.yaml`。

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
模型代码仍然只需要：

```python
import aten_size_profiler
```

输出报表包含 `summary`、`unique_shapes`、`detail`、
`input_tensors` 和 `metadata` 五个 Sheet。分布式运行时，每个 Global
Rank 会写入独立的 `*_rank<N>.xlsx` 文件：

```bash
torchrun --standalone --nproc_per_node=2 your_model.py
```

## 生命周期控制

可以使用 `pause_profiler()` 和 `resume_profiler()` 临时暂停和恢复采集。
`stop_profiler()` 会停止采集并导出一次报表。
如果不需要导出报表，使用 `stop_profiler(export=False)`，包括进程退出
时的兜底处理。

## 当前限制

当前实现面向 PyTorch Eager、标准 `nn.Module` 推理和标准单 Optimizer
训练。暂不承诺支持 `torch.compile`、复杂训练引擎、梯度累积、多个
Optimizer、自定义 iteration 边界、跨线程执行、跨节点分布式恢复或
Excel Sheet 自动拆分。

当任一 Sheet 可能超过 Excel 的 1,048,576 行限制时，Excel 导出会明确
失败。采集期间调用记录会保存在内存中，`logical_bytes` 不代表实际
GPU 物理显存占用。
