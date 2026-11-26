# SDR-005 环境与安装记录

| 检查项 | 实测或核实结果 |
|---|---|
| WSL2 Ubuntu 22.04 | Python 3.10.12；系统环境找不到 Torch、Triton、mamba_ssm、pip3 |
| WSL GPU | RTX 3090 24,576 MiB，驱动 610.88 |
| WSL 编译与空间 | GCC 11.4.0、nvcc 11.5.119；未发现 cmake/ninja；根卷约 93 GiB 可用 |
| 原生 Windows CUDA | 现有隔离 Python 3.9.21、Torch 2.5.1、CUDA runtime 12.4、torchvision 0.20.1、SciPy 1.13.1、Pillow 10.4.0；缺 Triton、mamba_ssm |
| 官方固定源 | Mamba v2.2.4，95d8aba8a8c75aedcaa6143713b11e745e7cd0d9，Apache-2.0 |
| 源码编译 | v2.2.4 的 setup.py 明确要求 nvcc ≥11.6，当前 WSL 工具链不足 |
| 下载与安装 | 安装 BLOCKED。缺少已批准且完整的 Linux CUDA/Torch/Triton/Mamba 本地依赖集合；未探测下载端点，未下载任何包，未修改宿主环境 |
| 官方 SSD 测试 | 原生 CUDA 条件测试因缺 mamba_ssm 跳过；FP32/BF16 官方结果均 NOT_RUN |
| 本轮 fallback | 原生 CUDA 上的 FP32 分块 oracle；仅用于本轮小规模真实数据验证 |

上游版本、许可证和源码限制通过公开文档或仓库元数据核对。候选 Linux 组合是 Python 3.10、CUDA 版 Torch 2.5.1、该 Torch release 的 Triton 3.1.0、官方 v2.2.4；本轮未安装或验证该组合。若获授权使用完整的已核验本地 wheelhouse，安装需仅从本地加载且锁定依赖哈希；任何 wheel 的版本、平台、CUDA、Torch ABI 和 SHA-256 必须与清单一致。以下是对已备好 wheelhouse 的离线执行形式，本轮未执行。

```bash
python3 -m venv "$SDR005_VENV"
"$SDR005_VENV/bin/python" -m pip install --no-index --find-links "$SDR005_WHEELHOUSE" --require-hashes -r "$SDR005_APPROVED_LOCK"
PYTHONPATH=src "$SDR005_VENV/bin/python" -m pytest -ra -s tests/test_sdr005.py tests/test_ssm.py -k official
```

[锁定算子](https://github.com/state-spaces/mamba/blob/95d8aba8a8c75aedcaa6143713b11e745e7cd0d9/mamba_ssm/ops/triton/ssd_combined.py) · [许可证](https://github.com/state-spaces/mamba/blob/v2.2.4/LICENSE) · [源码构建条件](https://github.com/state-spaces/mamba/blob/v2.2.4/setup.py) · [官方测试日志](gpu_official_parity_log.txt)

同一相关研究项目的本地 ResNet18 缓存名含官方哈希前缀 f37072fd。实测文件 10,157,520 字节，SHA-256 e6aabce200af3a929ec590217478bcc8ffafb1b9bdc310ff1a775bd4bbbe1e97，与前缀不符，默认缓存亦无对应文件。训练全部使用从头初始化的共同 CNN（权重哈希为空）。此文件未加载、上传或重新下载。
