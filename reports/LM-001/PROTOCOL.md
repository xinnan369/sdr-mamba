# LM-001 冻结协议与待授权预算

**CPU_ONLY_GPU_NOT_AUTHORIZED。** 本轮完成 CPU 模型、输入和评估接线。各行尚无 GPU 训练、精度或吞吐测量。[配置](../../configs/lm001.json) · [展开矩阵](experiment_matrix.json)

## 同协议对照

使用 LC-007B 的 SHB 360/40 清洗后分割，继续使用 ImageNet1K-V1 ResNet-18、P2/P3、192 通道投影、最近邻融合与相同 stride-4 双槽点预测头。代码从冻结的 LC-007G Git 对象临时读取，数据及权重从本地校验的 LC-007B 缓存读取。采用 LC-007B 的固定 8×8 P3 窗口，两次 shift 均为 0。LC-007G 的初始化、跨窗审计和资源计法用于核对。其四轮移位组合保留为历史。

| 组 | 窗口 mixer | 调用与块 | 监督 |
|---|---|---|---|
| CNN-only | 无 | 0 调用 | 仅 final |
| Shared-2 Transformer | 6 头窗口自注意力 | 同一块两调用 | final + .25 early |
| Unshared-2 Transformer | 同上 | 两个独立块各一调用 | 同上 |
| Shared-2 Mamba | 自定义门控 SSD 参考 mixer | 同一块两调用 | 同上 |
| Unshared-2 Mamba | 同上 | 两个独立块各一调用 | 同上 |

重复块保留原两处 LayerNorm、FFN 和残差，只替换 Mamba 组的注意力算子。共享组复用参数对象，独立组第一块与相应共享组同初值，第二块沿用 LC-007B 的独立初始化约定。关闭循环 embedding、阶段门控、密度损失、移位和梯度 checkpointing。训练使用原分辨率 512px 随机 crop、同一每图 crop 与 shuffle 序列、batch 2、4 workers、可训练 backbone 与训练态 BN。统一 BF16，参考 SSD 的递推矩阵以 FP32 计算。AdamW 峰值 LR .0002、WD .01、360 更新 warmup、cosine 最低 LR .00002。40 × 180 次成功更新，无早停与最佳 checkpoint 重选。CPU backward 只用生成的数据，正式优化更新为零。

每次成功优化更新后在唯一参数上更新 FP32 EMA，衰减 .99。live 与 EMA 共享当时 BN 缓冲。逐 epoch 对两种 policy 各做一次相同的验证 tile 扫描，循环组一次 tile 前向输出 early/final。tile 512、overlap 128，以 midpoint ownership 分配计数，阈值严格 >.30。Local nAP 使用原始帧内 score >.05 候选，按 LC 匹配器的稳定分数排序、归一化距离及 GT ID 打破平局。计数 TP/FP/FN 严格 δ=.50，AP 使用 δ=.05, .10, …, .50 的 pooled one-to-one 提取。密度边界只从训练计算，为 82.6517741 与 180.9861925 人/Mpx。主报告每种子 E36–40 final EMA 的 MAE 和 Local nAP，等权平均两种子。保留 live、E40、RMSE、偏差、FP/FN、密度分桶、逐图和阶段数据。对照优先同共享状态下比 mixer、同 mixer 内比共享与独立，CNN 是无 mixer 基准。官方测试推理为 NOT_RUN。[LC-007B 协议](https://github.com/xinnan369/looped-transform/blob/0aeaea0562f075fd28ecaa48c6400193a7cf22f4/reports/LC-007B/protocol.md) · [LC-007G 资源](https://github.com/xinnan369/looped-transform/blob/4a0f7b023e314af8ac6fe3c8ea0390d3428dee63/reports/LC-007G/resource_audit.md)

## GPU 预算提案

单 RTX 3090 串行十组，每组 7,200 次成功更新，共 72,000 更新。种子 42 依配置顺序，123 反向。LC-007B 八组曾实测约 4,043 秒，LC-007G 四组更深的移位实验约 3,610 秒。历史统计和环境不同，不能据此量出本轮 PyTorch SSD 的 GPU 速度。

| 项目 | 待授权上限或启动条件 |
|---|---:|
| 全轮累积同步 GPU 阶段 | 21,600 秒（6 小时），含预检 |
| 每个正式 run | 3,600 秒，受全轮余额约束 |
| 可选合成 BF16 预检 | 600 秒，零优化更新 |
| 开跑时设备可用显存 | 至少 8 GiB |
| 峰值 allocator reserved | 6 GiB |
| 额外数据、依赖与权重下载 | 0 字节 |

批准后先用 batch 2 的 512 合成输入核对相同后端上的前向、反向、显存和逐阶段耗时，以实测估计十组完整更新、双读出评估与存盘费用。预检不通过或预计超限时暂停，保留原协议与证据。正式运行记录同步阶段时间、峰值 allocated/reserved、成功更新、LR 与 crop 校验和、有限性、EMA 步数和已完成 checkpoint。超限或出现不一致时停止，保留原记录。GPU 启动入口与资源检查的首次运行待明确授权。
