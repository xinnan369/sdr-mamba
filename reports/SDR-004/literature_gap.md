# SDR-004 机制重合核查

核查日期 2026-11-26。比较一手论文及作者实现。“未见”限于所查材料。直接重合需同时满足三个条件：每个物理边界行或列保留独立 SSM 末态，该末态初始化下一共享外层循环的相邻窗同线，以及外层循环共享扫描参数。

| 工作与日期 | SSM 状态和空间顺序 | 权重共享、边界线与动态性 | 一手来源 |
|---|---|---|---|
| LocalMamba，预印本 2024-03-14；ECCV 2024 Workshops，论文集 2025-05-12 | 局部窗口扫描与全局方向候选，所查方案未见上一外轮窗末态初始化下游窗 | 各层搜索扫描组合，未见共享外轮或对齐末态库 | [arXiv](https://arxiv.org/abs/2403.09338) · [论文集](https://doi.org/10.1007/978-3-031-91979-4_2) · [作者代码](https://github.com/hunto/LocalMamba) |
| PVMamba，ICCV 2025（2025-10-19 至 25） | DSA 用局部或学习的空间采样并行构造隐藏状态。图 5 与 §3.3 的层复用对象是采样点 | 深层学习采样位置，浅层限制局部。未见逐线末态初始化或固定权重外循环 | [ICCV PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Xie_PVMamba_Parallelizing_Vision_Mamba_via_Dynamic_State_Aggregation_ICCV_2025_paper.pdf) · [IEEE](https://ieeexplore.ieee.org/abstract/document/11443324) · [作者代码](https://github.com/VISION-SJTU/PVMamba) |
| FractalMamba++，v1 2025-05-20，v4 2026-05-05 | v1 §III-C 的 CSR 将较早内部隐藏状态注入同一扫描中较晚的 patch。v4 的 FHSC 将快捷路径与 Hilbert 结构绑定 | **内部状态直接复用已有先例。** 所查机制在单条序列内跳接，未见跨共享外轮的逐物理边界线末态。++ 官方代码为 NOT_VERIFIED，未将 AAAI 前身代码视为 ++ 实现 | [版本页](https://arxiv.org/abs/2505.14062) · [v1 §III-C](https://arxiv.org/html/2505.14062v1#S3.SS3) · [v4 全文](https://arxiv.org/html/2505.14062v4) |
| VMamba，预印本 2024-01-18，NeurIPS 2024 | SS2D 用四条横纵展开序列，逆置及融合在视觉块内完成 | 四方向视觉扫描已有先例。所查 VSS 堆叠未见共享外轮与同线窗末态复用 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/baa2da9ae4bfed26520bb61d259a3653-Abstract.html) · [作者代码](https://github.com/MzeroMiko/VMamba) · [arXiv](https://arxiv.org/abs/2401.10166) |
| PlainMamba，预印本 2024-03-26，BMVC 2024（2024-11-25 至 28） | 空间连续 2D 路径和方向感知更新，状态在同一次序列内递推 | 空间连续扫描已有先例。所查块堆叠未见边界线初态或共享外循环 | [BMVC](https://bmvc2024.org/proceedings/755/) · [论文](https://arxiv.org/html/2403.17695) · [作者代码](https://github.com/ChenhongyiYang/PlainMamba) |
| 2DMamba，CVPR 2025（2025-06） | §3.2、图 2–3 先按每行独立水平扫描，再对每列独立扫描，以二维中间状态形成方向关联 | **独立行列和二维状态传递均已有先例。** 所查公式在同一块前向内完成，未见局部窗的外轮状态库 | [CVPR 论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_2DMamba_Efficient_State_Space_Model_for_Image_Representation_with_Applications_CVPR_2025_paper.pdf) · [作者代码](https://github.com/AtlasAnalyticsLab/2DMamba) |
| Mamba-ND，预印本 2024-02-08，ECCV 2024 | 多维轴依预定次序展开，逐层交替排列 | 按轴扫描已有先例。未见共享外循环的物理边界逐线末态 | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4811_ECCV_2024_paper.php) · [作者代码](https://github.com/jacklishufan/Mamba-ND) |
| QuadMamba，预印本 2024-10-09，NeurIPS 2024 | 学习四叉树划分，多方向窗口位移提供跨区信息 | 根据输入动态划窗，所查跨区机制没有固定共享外轮的同线末态库 | [NeurIPS](https://papers.neurips.cc/paper_files/paper/2024/hash/d592bfaedf2f0861d7084cceba208d18-Abstract-Conference.html) · [作者代码](https://github.com/VISION-SJTU/QuadMamba) |
| Looped SSMs，预印本 2026-05-15 | §3.1 的深度递归将同一 SSM 块反复作用于表示，比较完全或部分共享 | **固定权重 SSM 外循环已有先例。** 时间序列的块输出送入下一轮，所查公式未见额外跨窗末态。专用实现为 NOT_VERIFIED | [一手论文](https://arxiv.org/html/2605.16048v1) |
| SF-Mamba，预印本 2026-03-17，ICML 2026-07-06 至 11 | patch swapping 引入双向信息，batch folding 后周期重置状态 | 重置和序列重排已有先例，未见邻窗逐线末态跨共享轮次初始化。专用实现为 NOT_VERIFIED | [PMLR](https://proceedings.mlr.press/v306/yoshimura26a.html) · [预印本](https://arxiv.org/abs/2603.16423) |
| StateLoop，预印本 2026-10-26 | 循环 Transformer 维护每 token 的持久向量，按学习门读写 | 共享外循环及额外持久状态已有先例。状态是独立通道，不是物理窗的逐线扫描末态。官方实现为 NOT_VERIFIED | [arXiv](https://arxiv.org/abs/2610.35544) |

PVMamba 的会场 HTML 返回 403，核查使用会场 PDF 和作者仓库。其他版本及访问限制见表。已查材料未给出三条件同时成立的直接例子，无法据此确立首创。适合报告的区别是固定相同线扫描、容量和邻居后，测量物理同线与固定错配的差异。轴向扫描、内部状态重用或循环共享权重均不能单独作为创新主张。
