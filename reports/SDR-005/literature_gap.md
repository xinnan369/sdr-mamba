# SDR-005 端点复用的一手文献核查

核对重点是同一图像上四条轴向方向族共享外轮、同一物理行/列、上一轮反向真实末态初始化下一轮正向起点。按时间、层和共享外轮区分状态传递。以下结论只覆盖所查论文和实施说明。

| 工作 | 已有做法与同本轮的重合 | 一手来源 |
|---|---|---|
| **Arcee**，预印本 2025-11-14 | §2.3.2–2.3.3 已公开跨视觉 Mamba 块的内部末态到初态的可微交接，默认边界映射恒等，无新增参数。§3 已将其叠加到不同层间扫描次序及方向专用权重的骨干。这与 EASR 的一般状态交接机制**直接重合**。其通用边界映射可以表达固定置换。所查默认实验未独立比较共享四轴外轮中的同物理端点反向传递与 same-direction/wrong-line | [原文](https://arxiv.org/html/2511.11243) · [作者 CUDA/训练代码](https://github.com/JiteshChavan/rc2) |
| **MambaVLT**，预印本 2024-11-23，CVPR 2025-06 | 式 (6) 用上一个视频时刻真实末态与可学习初态的加权和初始化新的视觉跟踪扫描，同时采用双向扫描并部分共享方向参数。它已覆盖视觉 SSD/Mamba 状态跨次使用这一广义思想。状态跨的是视频帧，未见同一图像逐行列端点的共享外轮对照 | [CVPR](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_MambaVLT_Time-Evolving_Multimodal_State_Space_Model_for_Vision-Language_Tracking_CVPR_2025_paper.html) · [含式 (6) 的原文](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_MambaVLT_Time-Evolving_Multimodal_State_Space_Model_for_Vision-Language_Tracking_CVPR_2025_paper.pdf) |
| **VMambaCC**，预印本 2024-05-07 | 人群计数已经使用四方向 VSS、高层多头特征注意力和分层融合。层间主要共享或融合的是输出特征，所查方法未见上一轮反向内部末态按行列初始化 | [预印本](https://arxiv.org/abs/2405.03978) · [方法全文](https://arxiv.org/html/2405.03978) |
| **VSSD**，预印本 2024-07-26，ICCV 2025-10 | 通过同一层内的非因果 SSD 消除累积权重的幅度依赖，并与多方向扫描融合。它改变同轮计算核，所查方法未见固定共享外轮的末态端点映射 | [ICCV 原文](https://openaccess.thecvf.com/content/ICCV2025/html/Shi_VSSD_Vision_Mamba_with_Non-Causal_State_Space_Duality_ICCV_2025_paper.html) · [作者代码](https://github.com/YuHengsss/VSSD) |
| **2DMamba**，CVPR 2025-06 | 同层独立横向行扫描和纵向列扫描、二维状态组合以及空间连续性已有先例。所查公式不执行跨共享外轮的反向末态交接 | [CVPR 原文](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_2DMamba_Efficient_State_Space_Model_for_Image_Representation_with_Applications_CVPR_2025_paper.pdf) · [作者代码](https://github.com/AtlasAnalyticsLab/2DMamba) |
| **PVMamba**，ICCV 2025-10 | 采用 DSA 汇拢局部或动态采样点，层间 multiplexing 复用采样位置。状态参与空间混合，所查方法未将固定方向的真实末态带入共享外轮起点 | [ICCV PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Xie_PVMamba_Parallelizing_Vision_Mamba_via_Dynamic_State_Aggregation_ICCV_2025_paper.pdf) · [作者代码](https://github.com/VISION-SJTU/PVMamba) |
| **FractalMamba++**，v1 2025-05-20，v4 2026-05-05 | CSR/FHSC 已直接重用内部状态，路径位于一次 Hilbert 序列，沿几何位置跳接。它与“有结构的内部状态复用”重合，未见同轴反向端点的固定外轮 | [版本页](https://arxiv.org/abs/2505.14062) · [v1 CSR](https://arxiv.org/html/2505.14062v1) · [v4 FHSC](https://arxiv.org/html/2505.14062v4) |
| **Looped SSMs / Reshape and Recur**，v1 2026-05-15，v2 2026-10-01 | 反复作用同一 SSM 块到上一轮输出表示。固定外轮共权已有先例，所查架构未传递同物理轴的额外内部末态 | [版本记录](https://arxiv.org/abs/2605.16048) · [共享方案](https://arxiv.org/html/2605.16048v2) |
| **ZR²ViM**，2026-03-04 | §3.4 的 NRC 将局部输出线性投影到外状态。§3.5 的 CZ-WKV 把上一步 Bi-WKV 输出作为下一步 value，循环 zigzag 方向。递归方向与局部–全局重分配已有先例。它使用 RWKV 路径和上一步的输出表示，不等同于同一轴线的反向内部 SSD 末态输入 | [原论文 §3.4–3.5](https://pmc.ncbi.nlm.nih.gov/articles/PMC12996084/) |
| **Vision Mamba、Zigma**，2024 | 前者展示视觉 SSM 同轮双向方向，后者展示在不同层交替扫描。方向反转、多方向融合和层间次序变化已有先例。Arcee 已将末态交接叠加到相关生成骨干 | [Vision Mamba](https://arxiv.org/abs/2401.09417) · [Arcee §3.1–3.3 的 Zigma 对照](https://arxiv.org/html/2511.11243) |

原 VSSD 的 www HTML 入口返回 403，使用 CVF 官方规范域名。VMambaCC 与 ZR²ViM 的专用公开实现未核实。现有结果将本轮降级为**已知跨块状态复用机制的端点约束比较**。剩余可证伪问题限于 ShanghaiTech 上固定四轴共享循环内的 opposite-direction/同端点约束，是否优于 reset、same-direction、wrong-line 和等负载特征。即便得到正结果，Mamba 人群计数、双向扫描、状态交接或共享循环也不能作首创声明。
