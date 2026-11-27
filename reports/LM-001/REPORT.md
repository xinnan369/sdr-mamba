# LM-001 | 共同点预测平台 CPU 工程报告

LM-001 在 SDR-Mamba 仓库单独交付。参考 LC-007B 的公平双调用与 EMA 协议、LC-007G 的初始化与空间可达核验。MAR-001B 和 SDR 的已验收结果保留。五组均复用预训练 ResNet-18、P2/P3、192 维投影、融合及原 stride-4 双槽人头点预测头。在固定窗口块中，仅 Mamba 组以自定义 SSD 参考算子替换 attention，LayerNorm、FFN 和残差保持一致。具体训练、评估与预算见冻结协议。[冻结协议](PROTOCOL.md) · [矩阵](experiment_matrix.json) · [LC-007B 验收 PR](https://github.com/xinnan369/looped-transform/pull/20) · [LC-007G 验收 PR](https://github.com/xinnan369/looped-transform/pull/30)

## 本地输入和初始化

使用已存在的 WSL Python 3.11.11、Torch 2.2.2、torchvision 0.17.2，并隐藏 CUDA。源码通过只读 Git archive 从 LC-007G 的 `4a0f7b023e314af8ac6fe3c8ea0390d3428dee63` 临时导入。历史配置、初始化和 crop 轨迹校验使用 LC-007B 的 `0aeaea0562f075fd28ecaa48c6400193a7cf22f4`。参考工作树与分支保持原状态。参考版本未检出顶层许可证文件，交付依赖其本地 Git 对象，只在当前仓库提交适配器和新算子。权重、标注、图片和本地路径留在本地。全轮数据、权重与依赖新增下载为零。

| 本地输入 | 校验结果 |
|---|---|
| ImageNet1K-V1 ResNet-18 | 46,830,571 字节，SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| LC-007B SHB 清单 | SHA-256 `de60892266c08bce96a3ea07e884c86eec8d05e0a9d31c0949bcff9e925fe482` |
| 训练 / 内部验证 | 360 / 40 图，44,856 / 4,295 个清洗后头部标注 |

400 张图的文件可读、尺寸及唯一 SHA-256 均核对，和已经审计的原 SHB 训练人口一致。验证顺序与 LC-007B 完全相同。每个种子内五组公共参数和 BN 缓冲哈希一致。CNN 与两种 Transformer 的完整初始化逐种子复现 LC-007B，Mamba 的非 mixer 块张量与对应 Transformer 相同。共享与非共享各族第一块初始化相同，非共享第二块独立。两个种子的全 40 epoch crop 顺序和 7,200 步 LR 轨迹匹配 LC-007B 原记录。首轮直接映射权重时，历史权重缺少 BN 批次计数缓冲，触发 KeyError。已改用 torchvision 严格加载补齐历史缓冲，并通过完整状态哈希确认修复。[资产和初始化证据](cpu_audit.json) · [首轮失败日志](cpu_audit_initial_bn_failure.txt)

## 参数、FLOPs 与可达性

| 组 | 模型总参数（均可训） | 替换所在算子参数 | final FLOPs | early+final FLOPs |
|---|---:|---:|---:|---:|
| CNN-only | 1,151,750 | 0 | 25.095 G | 25.095 G |
| Shared-2 Transformer | 1,596,614 | 148,224 | 32.745 G | 47.278 G |
| Unshared-2 Transformer | 2,041,478 | 296,448 | 32.745 G | 47.278 G |
| Shared-2 Mamba | 1,593,150 | 144,760 | 32.420 G | 46.953 G |
| Unshared-2 Mamba | 2,034,550 | 289,520 | 32.420 G | 46.953 G |

以上为 CPU 上执行单个 512×512 tile 后按张量形状复算，用两次运算计一个 MAC。公共 backbone 与投影约 10.561 G FLOPs，每次点读出约 14.533 G。注意力计 Q/K/V、输出投影与两个注意力乘积。SSD 计输入/输出投影、每条线的 B/C 点积与每头状态读出矩阵乘。省略激活、归一化、指数和搬运，表中不是 GPU 延迟或实测训练成本。在相同共享条件下，两个 mixer 的全模型参数及 staged FLOPs 接近，但内部路径不同。

| 空间量 | Transformer | Mamba 参考 SSD |
|---|---|---|
| 单次 mixer 的范围 | 固定 8×8 窗口全注意力，64 个 token | 同窗逐行逐列四向扫描，单线最长 8 token |
| 窗内单个 token 的一次 / 两次理论可达 | 64 / 64 | 15 / 64 |
| 同一 512 tile 两次计算 | 每次 64 个窗口 | 32,768 个方向 token 访问 |
| 到其他 P3 窗口的 mixer 路径 | 无 | 无 |

窗口为 stride-8 P3 上的几何划分。边缘的无效 token 不读出、不影响 SSD 状态。双轮轴向两跳理论上覆盖全窗，CPU 梯度在缩小窗口上验证该路径和窗口间隔离。可达性针对 mixer，最终点头还读取公共 CNN 特征。按理论卷积几何，P2/P3 直接路径的名义感受野宽度为 43/99 px，忽略训练态 BN 的全局统计。点头最近邻融合后的 3×3 卷积可能在窗口边界接触相邻窗口，不能把 mixer 图等同于最终像素到预测的感受野。

## 参考算子与官方实现

Mamba 组使用单组 B/C、14 个状态空间头、每头宽 16、状态维 32，输入依赖的步长、负衰减、skip 和投影门控。实现以窗口内短因果矩阵形式计算 SSD 求和，对行列反转方向等权融合，状态按线、方向与调用复位。BF16 输入的核心求和升到 FP32。没有官方 Mamba-2 的因果卷积、门控 RMSNorm 和融合 Triton/CUDA 路径，也不宣称官方后端兼容。FP64 单测将参考算子的前向和全部 6 类输入梯度与显式逐步递推逐项比较，包含非满窗的 mask 及全无效行。另有 CPU BF16 前后向检查。官方实现的 CUDA 数值一致性、内存和速度为 NOT_RUN。[官方 Mamba-2 模块](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mamba2.py) · [官方融合算子](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_combined.py) · [参考 mixer](../../src/sdr_mamba/lm001.py)

## 验收与复现

两种子十个 arm 的 CPU 前向、点损失 backward 与合成图内部验证管线均通过。全套测试 80 项通过、3 项跳过。检查覆盖中间监督梯度路由、共享对象身份、非共享参数隔离、EMA 唯一参数与 live BN/RNG 恢复、严格计数阈值和独立 Local nAP 候选池。合成图仅用于验证工程接线，参数未做优化更新。正式 GPU 预检、GPU 训练、正式开发集精度和官方测试均为 NOT_RUN。预注册 CPU 实现提交为 `c47f66697829ce2faa1990ac7ce13e755e325c0d`。[CPU 审计](cpu_audit.json) · [运行日志](cpu_audit_log.txt) · [测试](test_log.txt) · [初次测试失败](test_initial_bn_failure.txt)

在已有 WSL 环境从当前仓库根目录运行。将下列变量指向现有本地缓存、已具备依赖的 Python 与参考仓库。结果写入新目录。

```bash
export CUDA_VISIBLE_DEVICES="" PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export LM001_REFERENCE LM001_ASSETS LM001_PYTHON
"$LM001_PYTHON" -m scripts.prepare_lm001 --reference "$LM001_REFERENCE" \
  --manifest "$LM001_ASSETS/shanghaitech_b.jsonl" \
  --weights "$LM001_ASSETS/torch/hub/checkpoints/resnet18-f37072fd.pth" \
  --output runs/lm001-cpu
"$LM001_PYTHON" -m pytest -ra -s
```
