# SDR-006 预注册

从 PR #8 的 `6e5f53f` 启动。一次基础 Looped SSM 可行性审计，冻结 EASR、LAST、BCST、SST 变体和 router。官方 316 图只做标注与完整性审计，模型不读取该分割。

原论文 Table 1 对 Part B 给出 88,488 人。此协议以 MATLAB `image_info.number` 为完整图计数，逐文件严格核对 `location` 行数。DM-Count 作者的 ShanghaiTech 验证入口也使用完整点数组行数。本地相关项目 LoopCrowd 的几何约定使用原始坐标、半开矩形及边界过滤，适用于严格在帧内点位。没有找到发布端明确记录的像素原点，故空间主分析采用原坐标与半开矩形，有限越界点逐轴投影到合法边界，让密度质量保持官方全图点数。偏离图像较远的点仍计入，空间落点只代表边界近似。另量化减 1 和直接过滤两种敏感性。此坐标选择只影响空间分配与训练 crop，主全图计数由原始标注固定。[原数据发布页](https://github.com/desenzhou/ShanghaiTechDataset) · [原论文](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf) · [DM-Count 评估实现](https://github.com/cvlab-stonybrook/DM-Count/blob/master/datasets/crowd.py#L923-L942)

冻结 SDR-005 的 60 张开发图。旧训练 340 张按原始总数排序，在每个 17 图分层固定抽 3 张，得到 60 张内部确认图与 280 张新训练图。确认图曾用于历史训练，本次所有模型从头初始化，无旧权重。承接已验证 FP32 CUDA 分块 oracle，继续使用共同 scratch CNN、stride 8 和正值密度头。没有可信可用的统一预训练权重，原研究用缓存未通过源哈希核对。

每个种子各组 2,000 更新，batch=2。固定八步循环含 4 次 256 crop、3 次 512 crop、1 次原分辨率 768×1024。两张同尺寸图同批，均匀有放回按图采样，原图无 resize，随机水平翻转。所有组共享训练数据次序。每 200 更新在旧开发集以完整图评估。统一 AdamW 提升到 0.0003，150 步 warmup，余程余弦降至 0.00003，权重衰减 0.01，梯度裁剪 1。目标为保持质量的平滑密度回归与 log1p 计数损失，与 SDR-005 v3 的系数相同。训练前六个真实固定 crop 检验需同时降低损失和计数 MAE，且 MAE 低于全零基线至少 20%。

F0 共同 CNN，F1 四轮共享轴向 reset，F2 四个从 F1 初始块拷贝的独立轴向 reset，F3 四轮共享 depthwise 3×3＋pointwise 残差块，F4 独立训练的一轮共享轴向 reset。F1/F2 同 token 和轴向块定义。F3 参数近似匹配，算量差异披露。所有训练均以本组深度末轮指标反传。

从十个开发候选 checkpoint 中按最小 MAE、GAME1、步数顺序选择。主指标全图，确定性无重叠 256 分块只诊断。末步和最后三点均值、样本 SD、相对 SD、相对均值最大幅度均保留。seed 0 的通过条件是 F1 相较 F0 的 MAE 降低至少 3%，相较 F2 的 MAE 不劣于 3%，GAME1 对两者不回退超过 3%，五组最后三个 MAE 的相对 SD 均有限且 ≤15%。通过后补齐 seed 1/2，确认集一次解锁。最终门使用 Issue #9 的三种子逐组阈值。每组 2 GPU 小时，全轮 12 GPU 小时。配置和数据审计先提交。
