# SDR-006 环境和已知限制

原生 Windows 环境使用已有隔离 Python 3.9.21、Torch 2.5.1 / CUDA 12.4、NVIDIA RTX 3090（24 GiB），驱动 610.88。预检时设备空闲约 18,981 MiB。本轮 F1、F2、F3、F4 以及基础 F0 在真实 256、512 和完整 768×1024 图上均运行了前向、非零梯度、梯度裁剪和优化步。完整双图的最大 PyTorch 峰值为约 2.24 GiB（F1）。[逐组预检](preflight.csv) · [正式环境](environment.json)

本轮重新核查 WSL 当前环境。Python 3.10.12，根卷余 93 GiB，CUDA 编译器 nvcc 11.5 系列，系统环境找不到 Torch、Triton 或 mamba_ssm。固定上游 Mamba v2.2.4 要求 nvcc ≥11.6。原生 CUDA 的官方 SSD 条件测试再次确认没有 mamba_ssm，三项跳过。官方 SSD 安装状态 `BLOCKED`，FP32、BF16 一致性均 `NOT_RUN`。沿用上一轮完成显式递推对照的 CUDA FP32 chunk=16 oracle。实验结果限于该研究算子，生产实现延迟仍未验证。[本轮 CUDA 条件测试](official_parity_log.txt) · [上一轮算子对照](../SDR-005/gpu_chunked_parity_log.txt) · [上游构建条件](https://github.com/state-spaces/mamba/blob/v2.2.4/setup.py)

本轮复用完整本地 Part B 和已有隔离依赖。未下载依赖、数据或权重，未修改 Python 全局环境或 GPU 驱动。上一轮 ResNet18 缓存的实际 SHA-256 与注册来源前缀不一致。本轮全部模型使用同一 scratch CNN 构造，没有加载缓存权重或旧训练 checkpoint。预检测量发生于预注册提交前，预检环境记录中的提交号仅标识当时的检出基线。[上一轮权重核对](../SDR-005/installation_log.md) · [预检环境](preflight_environment.json)
