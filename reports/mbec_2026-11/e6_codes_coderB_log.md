# E6 编码日志 (Coder B)

- 输入: `e6_frame.csv` 中 `decision = INCLUDED` 的 37 篇; 全文位于 `e6_fulltext/`。
- 独立性: 文件列表已在 Python 中过滤掉名称含 coder/coderA 的文件; 未打开任何 coderA 文件。只写入两个输出文件, 未提交 git。
- 输出: `e6_codes_coderB.csv` (37 行, 全部 rank 覆盖), 本日志。
- 临时文件目录: 本机临时目录(未随仓库发布)

## 1. 文本提取方法

- PDF (共 26 篇): `pypdf` 逐页提取 (python `extract.py`), 文本中带 `[[PAGE n]]` 页码标记。部分 PDF (如 rank 49 CLISA、rank 95、rank 122) 出现逐词断行, rank 49 另外用脚本把断行合并成连续文本后做上下文窗口检索。
- .txt (共 11 篇, 来自出版社 HTML 保存): 直接使用。少数行过长, 关键词检索会显示 "Omitted long matching line", 因此对相关段落用 Read 按行范围读取。
- `.source.html / .source.xml` 原始文件未使用: 所有 .txt 均包含方法、实验与结果表, 未判定为截断。
- 对 rank 24 (Brain Informatics .txt): 读取了 4.1–4.5 与表注, 但第 3 节方法部分仅做了关键词检索, 未逐段通读。

## 2. 每篇阅读范围 (rank: 文件)

逐篇均做了全文关键词扫描 (epoch / early stop / validation / best / checkpoint / hyperparameter / grid / tuned / normali* / standardi* / z-score / LOSO / target / unlabeled / github / available at / learning rate / test set / sensitivity), 再对数据集、协议、训练设置、结果表注与命中段落用 Read 或脚本窗口逐段阅读。下列为主要读取的章节:

| rank | 文件 | 主要阅读章节 |
|---|---|---|
| 1 | r01 (pdf 10-3390-e27090986) | 2.3.4 跨被试数据处理与 subject-independent 评估, 3.1–3.2 |
| 2 | r02 (pdf 10-1922-pjs) | 3C 微调 (超参数选择), 4 实验, CV/CSV 协议 |
| 4 | r04 (pdf access-2026-3692665) | IV-A 消融, IV-B 混合被试, IV-C 跨被试 (LOSO, 归一化) |
| 11 | r11 (txt s44163) | 3.1 数据与预处理, 3.2.4 融合与检查点, 3.4 实现细节, 4.1 |
| 12 | r12 (pdf fnhum-2024-1471634) | 3.1 数据集与设置, 算法 1, 超参数表, LOSO |
| 14 | r14 (pdf fnins-2026-1846592) | 5.2 四种验证协议, 5.3.1, 5.4.5 参数敏感性 |
| 17 | r17 (pdf access-2025-3574515) | IV-A SEED, IV-B 实现, IV-C 场景, IV-D 结果 |
| 23 | r23 (pdf tst-2026) | 4.1 实现, 4.2 超参数敏感性, 4.3 对比 (LOSO) |
| 24 | r24 (txt Brain Informatics) | 4.1–4.5 (数据、基线、结果、解释、消融); 第 3 节仅关键词 |
| 27 | r27 (pdf DAGAM) | 4.1 实现, 4.3 LOOCV |
| 30 | r30 (pdf LDG) | Training protocol, Parameter setting, Inter-subject DG, 收敛, 跨数据集 |
| 34 | r34 (pdf EDANN) | III-B 预处理, III-C 数据, IV-C 实验设计, IV-D 结果, 迭代图注 |
| 42 | r42 (txt Frontiers Psychiatry) | DANN 实现段, LOOCV 段, 数据集段 |
| 43 | r43 (pdf DFF-Net) | 4.2 实验设置, Table 2 超参数, 基线 |
| 45 | r45 (txt BiHADA) | LOSO 段, 训练段, 4.1, 4.5 敏感性, 预处理段 (-o 检索) |
| 49 | r49 (pdf CLISA) | 4.4 实现 (early stopping, 标准化), 4.5 协议, 验证数据来源检索 |
| 50 | r50 (txt diagnostics) | 2.3–2.10, 3.1, Table 4 (epochs) |
| 53 | r53 (txt MS-AGDA) | 实验 (LOOCV 段), 训练段, 迭代图注 |
| 64 | r64 (txt GA-PDPL) | 2 (GA 适应度), 4 实验协议 |
| 66 | r66 (pdf SSAS) | 3.2 实现 (epochs), 4.1, 4.5 敏感性, 代码链接 |
| 69 | r69 (pdf SH-MDA) | 2 预处理, 3 训练与表 3, 4.1 LOSO, 代码链接 |
| 82 | r82 (pdf EEG-DML) | IV-A 预处理, IV-C 实验设置, IV-G 超参数分析 |
| 89 | r89 (pdf BSR) | 方法概述, 结果表 2–3, LOSO 段 (文本较少) |
| 90 | r90 (pdf TNSRE CSDS) | V-A 参数估计, V-C 设置, V-D 结果 |
| 95 | r95 (pdf Sci Rep CSCL) | 方法 (DANN), 预处理, LOSO 与 10 折段, 表 2 |
| 102 | r102 (pdf TNSRE 2025) | III (EEG-CutMix), IV, V-A LOSO, Table I 标题, 代码链接 |
| 113 | r113 (txt BFE-Net) | 2.3.1 数据划分, 2.3.2 网络参数, 3.1–3.2 |
| 116 | r116 (pdf JBD) | 4.2 数据划分, 4.3 训练 (Table 2), 5.3 超参数 |
| 117 | r117 (pdf MBE SSA-RF) | 2 (SSA-RF 适应度), 3.3–3.4 实验协议 |
| 121 | r121 (pdf JBHI M3D) | IV-B 实验设置 (分类器组合选择), IV-C 协议, V 结果 |
| 122 | r122 (txt DSP-MCF) | 2.2 DG 定义, 4.1 评估 (LOSO), 4.2 实现 (save-best), 4.3 |
| 130 | r130 (txt CNN-KAN) | 预处理, 实现 (10 折 CV), 消融, 单被试实验段 |
| 131 | r131 (pdf RPGCN-GDA) | 3.4 GDA, 4 实验协议与训练设置 |
| 146 | r146 (pdf MS-DCDA) | 4.1–4.3, 5 结果 (ACC-best 定义), 5.1 |
| 150 | r150 (txt SCVAE-Net) | 4 实验设置, 敏感性段, 代码链接 |
| 161 | r161 (pdf HEDN) | IV 实现, V-C 协议与收敛稳定性, V-D, Table VIII |
| 168 | r168 (pdf EMO-CARE) | Evaluation and Training Protocol, 超参数优化, 代码链接 |

## 3. 不可用或部分可用的文本

- 无完全不可用的全文。
- rank 49 (CLISA, PDF): 文本逐词断行, 经脚本合并后可读, 但 Table S 等补充表未检查。
- rank 89 (BSR 2025): 文本较简略, 表格提取不完整, 数据集与 Table 3 的对应关系不明; 编码置信度为 LOW。
- rank 17 (access 2025): PDF 中部分公式与表格 OCR 错乱, 但 IV 节文字可读。
- rank 24 (txt): 第 3 节方法未逐段阅读 (仅关键词检索)。
- rank 113 (txt): 文本可读, 但其 "validation" 与 "testing dataset" 术语前后不一致, 已在 notes 中说明。

## 4. 编码约定与判断 (供 PI 审阅)

1. V1 = N 的论文 (rank 130) 按规程将 Q1–Q7 全部记为 NA。该文标题与正文使用 "cross-subject", 但 SEED 的评估为合并被试后的随机 10 折 CV, 未报告留出被试协议。
2. V1 = UNCLEAR 的 rank 24: 表 1 标注为 subject-independent, 但方法段明确为 8:2 随机划分, 因此不计为 Y。其余字段仍按规程填写。
3. 配置/超参数由测试或目标数据上的准确率、适应度或 "highest accuracy" 选择时, Q4 记 TEST_OR_TARGET。对于以测试集误差为适应度的 GA/SSA 优化 (rank 64, 117), 以及以目标被试准确率选择分类器组合的 rank 121, Q1 记为 BEST_ON_TEST (选择的是配置而非 epoch, 属于同类处理)。此为本编码的主观判断, 建议统一审阅。
4. rank 146 与 rank 161 的 Q1 = BEST_ON_TEST: rank 146 报告的是三轮 LOO 中目标域最高准确率 (ACC-best); rank 161 的主表 SEED 值 (94.00) 与 Table VIII 的 "Best" 列一致 (最终值为 84.17), 因此推断报告的是训练过程中的最佳值。后者为推断, 置信度 MEDIUM。
5. rank 45 Q1 = FIXED_LAST: 文中 "The accuracy was recorded at the end of each experiment" 且训练 200 epochs, 未描述选择规则。
6. rank 50 Q1 = FIXED_LAST: 原文 "we chose to train each model on a fixed number of epochs" 且放弃早停。Q4 记 TEST_OR_TARGET, 依据为 "The GNN+Transformer architecture was selected as the baseline" (依据 LOSO 结果表); 超参数网格在 "training data" 上评估, 不属于 validation 也不属于 test, 按规程记为 NOT_STATED 亦合理, 已在 notes 中说明。
7. rank 113 Q1 = NOT_STATED (LOW): 早停依据 AUC 阈值 0.99, 但监控的数据未说明; 最终报告 "average validation accuracy" 而划分中测试被试被称为 testing dataset。
8. rank 122 Q1 = VALIDATION, Q2 = UNCLEAR: "save-best" 基于 validation accuracy, 但验证集来源未说明。
9. rank 116 Q1 = VALIDATION, Q2 = SOURCE_SUBJECTS: 依据为 "The training and validation sets are from the source domain" 与 "performs best as our final choice" (选择指标未明说为验证集, 为推断)。
10. rank 168 与 rank 49: rank 168 的验证集为训练被试的 20%, 置信度 HIGH; rank 49 早停依据 "validation accuracy", 来源未说明, 因此 Q2 = UNCLEAR。
11. rank 30 的 Q4 为冲突陈述: 正则化参数按源域子域做 p 折 CV (VALIDATION 式), 但 k 的选择 "report the top-one recognition accuracy from the best parameter configuration" 指向测试准确率。按后者记为 TEST_OR_TARGET, 置信度 MEDIUM。
12. Q5 scope: rank 23 (electrode-wise, source and target) 与 rank 69 ("normalize them by different channels") 无法确定统计量范围, 记为 OTHER_OR_UNCLEAR; rank 34 的 Z-score 范围未说明, 同样记为 OTHER_OR_UNCLEAR。rank 4 为 "applied individually to each participant's data" (PER_SUBJECT); rank 122 与 rank 130 的归一化/标准化段落未发现数据级标准化。
13. Q6 (未标记目标数据参与训练): 对 DG 方法 (rank 30, 122) 记 N。rank 89 因方法描述不足 (DANN/CORAL 描述为 "source and target subjects") 记 UNCLEAR。rank 34 基于 "aims to accurately predict the labels of the target domain using the data from the source domain" 推断目标标签未用, 记 Y (MEDIUM)。
14. Q7 代码链接: rank 66 (github.com/liuyici/SSAS), 69 (github.com/Xitsuka/SH-MDA), 102 (文中为 "GitHub-seizeall/mycode", 无协议前缀, 按原样记录), 150 (github.com/braverSheep/SCVAE-Net), 161 (github.com/qwangwl/HEDN), 168 (github.com/Yegi03/EMO-CARE, PDF 断行后拼接)。其余为 N; rank 50 与 rank 64 文中的 GitHub 字样分别为 PMC 页面模板文字和数据集链接, 未计为代码链接。

## 5. 各代码的频数 (37 篇)

- V1: Y 35; UNCLEAR 1 (rank 24); N 1 (rank 130)
- Q1 (checkpoint_selection): NOT_STATED 24; VALIDATION 5 (rank 11, 49, 116, 122, 168); BEST_ON_TEST 5 (rank 64, 117, 121, 146, 161); FIXED_LAST 2 (rank 45, 50); NA 1 (rank 130)
- Q2 (validation_source, 仅 Q1=VALIDATION 时适用): SOURCE_SUBJECTS 3 (rank 11, 116, 168); UNCLEAR 2 (rank 49, 122); NA 32
- Q3 (epochs_stated): Y 20; N 16; NA 1 (rank 130)
- Q4 (hyperparameter_selection): TEST_OR_TARGET 17; NOT_STATED 15; VALIDATION 2 (rank 49, 168); PRIOR_OR_FIXED 2 (rank 90, 122); NA 1 (rank 130)
- Q5 (normalization_stated): Y 15; N 21; NA 1 (rank 130)
  - Q5_scope (仅 Q5=Y): PER_SUBJECT_OR_SESSION 10; TRAINING_STATISTICS 2 (rank 11, 102); OTHER_OR_UNCLEAR 3 (rank 23, 34, 69)
- Q6 (uses_unlabeled_target_in_training): Y 20; N 15; UNCLEAR 1 (rank 89); NA 1 (rank 130)
- Q7 (code_link): Y 6; N 30; NA 1 (rank 130)
- confidence (Q1): HIGH 2 (rank 11, 168); MEDIUM 33; LOW 2 (rank 89, 113)

注: 上述频数来自 CSV 中的最终值。V1=N 的 rank 130 其余字段为 NA, 故 Q 系列的合计并非 37。

## 6. 较难编码的论文

- rank 24 (V1 UNCLEAR; 标题称 cross-subject 但划分为随机 8:2)
- rank 130 (V1 N; 标题称 cross-subject 但为随机 10 折)
- rank 30 (Q4 冲突陈述)
- rank 49 (早停验证集来源不明)
- rank 50 (超参数基于训练数据网格; 架构选择基于 LOSO 表; Q4 归类存在争议)
- rank 113 (早停监控数据与 validation 术语不一致, 置信度 LOW)
- rank 122 (save-best 验证集来源不明)
- rank 89 (文本简略, 置信度 LOW, Q6 UNCLEAR)
- rank 161 (Q1 BEST_ON_TEST 为由表值推断)
- rank 64, 117, 121 (配置/适应度选择按测试集处理, 与 Q1 的 "epoch" 定义存在偏差, 需统一规则)
- rank 14 与 rank 45 (固定 500 / 200 epochs 是否应记为 FIXED_LAST: rank 14 未明确最终模型, 记 NOT_STATED; rank 45 有 "recorded at the end" 表述, 记 FIXED_LAST)
- rank 17 ("final model accuracy" 但无迭代数, 记 NOT_STATED)
