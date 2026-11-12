# 编码员 A 日志 (e6_codes_coderA)

## 阅读方式
- 输入: e6_frame.csv 中 decision=INCLUDED 的 37 篇; 全文取自 e6_fulltext/。未接触任何含 coderB/coder_B 的文件。
- PDF (26 篇): 用 eeg_venv 的 `pypdf` (`PdfReader.extract_text`) 逐页提取, 加 `[[PAGE n]]` 标记, 存于 scratchpad/coderA/。
- .txt + .source.html (11 篇): 直接读 .txt (均为 Springer/MDPI/Frontiers/PLOS 的完整正文, 含方法与实验节; 对 10-1186-s40708-024-00245-8 另核对 .source.html 长度一致, 全文仅约 2.7 万字符, 确为短文, 未截断)。
- 读法: 对每篇先用关键词扫描 (epoch, early stop, validation, best, checkpoint, model selection, hyperparameter, grid search, tuned, normaliz/standardiz/z-score, leave-one-subject-out/LOSO, target domain, unlabeled, github, code, learning rate, iteration, batch size 等, 并去掉参考文献) 取上下文片段通读, 再对数据集/实验协议/实现细节/超参/结果表题注所在段落用正则做定点全文读取; 实验设置小节基本完整读过。方法推导公式部分未逐字精读。
- 个别图像表格: 10-1109-tnsre-2025-3603190 的 Table I (超参设置) 在文本提取中缺失, 用 Read 读取 PDF 第 7 页图像核对 (无 epoch 数)。
- 两处技术问题: (1) 初次提取的两条记录因 stdin 编码 (gbk) 把希腊字母写坏, 已按原文重写 (14、45、69 的 Q4/Q5 引文及 90 的整条记录); (2) 超过 25 词的引文在输出时截取前 25 词并以 [...] 标记。

## 文本不可用/不完整的论文
- 无完全不可用的全文。
- 表格内容在文本提取中丢失(图像表): 82 (10.1109/access.2024.3458833, Table 3 超参及其调整范围缺失), 102 (Table I, 已用图像核对)。
- 89 (10.54392/bsr2513): 全文极简且文字被机器改写(如 "DAP, Seeds, Vesad"), 几乎无训练细节, 方案是否在 SEED 上做 LOSO 无法确定。
- 12/43 (Frontiers PDF) 数字以 `/one.tnum` 字形名编码, 文字可读但数字需拼读。
- 49 (CLISA): PDF 提取字符间有多余空格; 补充材料 (Table S1/S2) 不可得。
- 66、146: 提取文本缺空格(预印本排版), 可读。

## 汇总 (n=37)
V1 cross_subject_on_seed: Y 34, UNCLEAR 2 (24, 89), N 1 (130)
Q1 checkpoint_selection: NOT_STATED 29, VALIDATION 5 (11, 49, 116, 122, 168), BEST_ON_TEST 2 (121, 161), FIXED_LAST 0, NA 1 (130)
Q2 validation_source: SOURCE_SUBJECTS 3 (11, 116, 168), UNCLEAR 2 (49, 122), NA 32
Q3 epochs_stated: Y 20, N 16, NA 1
Q4 hyperparameter_selection: NOT_STATED 15, TEST_OR_TARGET 10, PRIOR_OR_FIXED 6, VALIDATION 5, NA 1
Q5 normalization_stated: Y 14, N 22, NA 1; scope: PER_SUBJECT_OR_SESSION 9, TRAINING_STATISTICS 2, OTHER_OR_UNCLEAR 3, (WHOLE_DATASET 0)
Q6 uses_unlabeled_target_in_training: Y 21, N 15, NA 1
Q7 code_link: Y 7 (66, 69, 102, 113, 150, 161, 168), N 29, NA 1
confidence (Q1): HIGH 16, MEDIUM 17, LOW 4 (89, 113, 116, 122)

## 编码约定 (供与另一编码员对照)
- V1=N 的 130 号: Q1–Q7 全部填 NA, 参考信息写在 notes。
- V1=UNCLEAR (24, 89) 仍照常编码其余项。
- Q1 仅在文字明确说明报告哪个检查点/如何选择时才编 FIXED_LAST/VALIDATION/BEST_ON_TEST; 只给出 epoch 数而未说明选哪个 epoch 一律 NOT_STATED (含只写 "fixed at 500 epochs" 而未明说取末轮的 14 号)。
- 非神经网络方法 (1, 64, 90, 117, 30, 121) 无 epoch/检查点概念: Q1 编 NOT_STATED (121 例外, 因明说按目标域最高准确率选择分类器/集成组合, 编 BEST_ON_TEST)。
- Q4: 论文只给出"设为某值"或"沿用前人"编 PRIOR_OR_FIXED; 有敏感性分析/网格搜索且以 LOSO/目标域准确率展示 "optimal/best" 并据此给出所用设置, 但未说明用的验证集时, 编 TEST_OR_TARGET (66, 69, 113, 116, 121, 150, 161 为此类判断; 64、117、23 为文中明说在 test/target 准确率上优化)。27、45 仅有敏感性分析而未说用其选值, 编 PRIOR_OR_FIXED。
- Q5 scope: 采用每个被试/会话自身统计量的编 PER_SUBJECT_OR_SESSION; 仅写 "normalized/standardized" 未说统计量来源的编 OTHER_OR_UNCLEAR; 网络内 BatchNorm/LayerNorm 不算数据归一化。
- Q6: 域适应/对抗中使用目标被试 (无标签) 数据参与训练编 Y; 领域泛化(明确不使用目标数据)编 N; 49 号在预测时用测试被试在线更新归一化统计量, 训练期无目标数据, 编 N 并在 notes 说明。

## 难以编码的论文 (逐条理由见 CSV notes)
- 113 (brainsci14030271): 以 "平均 AUC 达 0.99 即停止并用最后保存的权重", 但监控数据 (训练/留出被试) 未说明; 同时将 "average validation accuracy" 作为 LOSO 最终性能。
- 122 (fnhum.2026.1723907): 采用 "save-best / 最高 validation accuracy", 但 LOSO 协议只描述训练集与测试被试, 未描述验证集, 验证来源无法确定。
- 116 (s40537-026-01584-5): "select the model that performs best", 评价数据未说明; 仅因划分中明确有来自源域的验证集才编 VALIDATION。
- 161 (HEDN) 与 146 (MS-DCDA): 161 的主表数值等于其 Table VIII 的 "Best", 但正文未明说主表用最佳轮次; 146 报告 ACC-best (三轮 LOO 中最优), 轮次含义不明, 未按 BEST_ON_TEST 编。
- 50 (diagnostics16010144): 方法节称固定 epoch 不做早停, 结果节又称 "Early stopping was employed", 自相矛盾; 超参搜索 "on the training data" 但又 "again LOSOCV"。
- 49 (CLISA): 早停用 "validation accuracy", 验证集来源未说; 超参部分经验设定、部分交叉验证、部分 "chosen parameters ... top performances (Table S1,S2)"。
- 24 (s40708-024-00245-8) 与 89 (bsr2513): V1 无法确定 (24: 标称 subject-independent 但只写 8:2 随机划分; 89: 仅泛称 LOSO)。
- 2 (pjs 2026): CSV 调参 criterion 为验证准确率, 但 CSV 受试者留出时的调参所用数据不清。
- 30 (fnins 2023, LDG): Q4 同时有源域 p 折交叉验证与 "report the top-one accuracy from the best parameter configuration"。
- 130 (PLOS ONE): 标题称 cross-subject, 实际为合并数据随机 10 折, 编 V1=N。
