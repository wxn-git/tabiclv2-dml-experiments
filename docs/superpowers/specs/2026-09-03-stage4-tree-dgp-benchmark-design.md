# Stage 4 树状 DGP 家族与小样本高维压力实验设计

## 1. 研究目标

Stage 4 检验 TabICLv2 在真正的轴对齐树状数据生成过程（DGP）下，能否在双重机器学习（DML）中稳定优于公平调参的 XGBoost。实验同时包含两个互补面板：

1. **面板 A：标准树状 DGP 家族。** 比较不同但常见的树结构，判断结论是否依赖单一公式；
2. **面板 B：小样本高维树状 DGP。** 保持真实函数不变，只减少样本量、增加无关变量，判断 TabICLv2 的相对优势是否集中在数据稀缺环境。

本阶段不以“必须找到 TabICLv2 获胜的 DGP”为验收条件。若没有候选通过独立确认，应据实报告 TabICLv2 在标准分段常数树结构下不能稳定超过专门的树提升模型。

## 2. 与已有实验的关系

- 原 `tree` DGP 包含 `I(X2*X3>0)` 的纯 XOR 交互和 `I(X0+X1>0)` 的斜边界，保留为复杂表示压力测试，不作为标准树基准；
- `tree_simple` 由轴对齐阈值组成，纳入面板 A 的加性树桩结构；
- 已完成的 `tree_simple, n=2000, p=10` 结果仅作为先验证据，不与 Stage 4 的全新种子结果合并；
- Stage 1、Stage 2、Stage 3B 和 tree-simple Stage 3B 的代码、缓存和结果保持只读。

## 3. 共同的部分线性模型

所有配置使用：

```text
D = m0(X) + V
Y = theta0 * D + g0(X) + epsilon
theta0 = 1
X_j iid ~ N(0, 1)
V iid ~ N(0, 1)
epsilon iid ~ N(0, 1)
```

`V`、`epsilon` 和 `X` 相互独立。与现有代码一致，先计算 `raw_m` 和 `raw_g`，再分别中心化并缩放到总体样本标准差为 1：

```text
m0 = unit_scale(raw_m)
g0 = unit_scale(raw_g)
l0 = theta0 * m0 + g0
```

Stage 4 主实验固定噪声方差，不通过降低 `Var(V)` 放大 DML 对处理模型误差的敏感性。噪声强度只允许在主实验结束后作为预先另立的敏感性分析。

## 4. 三类轴对齐树结构

所有阈值固定为 0，使二叉分支在总体上近似平衡。正式 DGP 不包含乘积特征、变量和阈值的线性组合、纯 XOR 或向学习器提供的人工真值特征。

### 4.1 S1：加性树桩 `tree_stumps`

```text
raw_m = 0.9 I(X0 > 0) - 0.7 I(X1 > 0) + 0.5 I(X2 > 0)
raw_g = 0.8 I(X0 > 0) + 0.6 I(X3 > 0) - 0.5 I(X4 > 0)
```

该结构与现有 `tree_simple` 等价，用作最简单的标准树基准。`X0` 同时影响处理和结果，形成可由 DML 控制的混杂。

### 4.2 S2：标准层级树 `tree_hierarchical`

定义：

```text
H(r, left, right; a, b, c)
  = a I(Xr > 0)
  + b I(Xr > 0, Xleft > 0)
  + c I(Xr <= 0, Xright > 0)
```

使用：

```text
raw_m = H(0, 1, 2; 0.8, 0.6, -0.4)
raw_g = H(0, 3, 4; 0.7, 0.5, -0.4)
```

它可由一棵根节点为 `X0`、左右子节点分别为另一个变量的深度 2 回归树精确表示。对于独立对称变量：

```text
E[raw_m | X0 > 0]  = 1.1
E[raw_m | X0 <= 0] = -0.2
E[raw_g | X0 > 0]  = 0.95
E[raw_g | X0 <= 0] = -0.2
```

因此根节点具有明显的总体分裂收益，不是 XOR。

### 4.3 S3：浅层森林 `tree_forest_sum`

使用两棵深度 2 小树之和：

```text
raw_m = H(0, 1, 2; 0.55, 0.40, -0.30)
      + H(3, 4, 5; 0.45, -0.35, 0.30)

raw_g = H(0, 6, 7; 0.50, 0.35, -0.25)
      + H(3, 8, 9; 0.40, -0.30, 0.25)
```

该结构仍是纯轴对齐、分段常数函数，但需要学习多个根节点和条件分支。`m0` 与 `g0` 共享根变量 `X0` 和 `X3`，子节点不同。

## 5. 防止隐藏 XOR 的结构验收

每个 DGP 必须通过自动化总体近似检查后才能进入 smoke：

1. 使用至少 200,000 个无噪声样本近似每个真实根分裂前后的平方误差下降；
2. 每个定义中的真实根分裂必须具有严格为正、且大于数值容差的总体收益；
3. 对 S2 验证深度 2 真值树可将无噪声结构 MSE 降到数值精度范围；
4. 对 S3 验证两棵深度 2 真值树之和可重构 `raw_m` 和 `raw_g`；
5. 检查代码中不存在 `Xj*Xk > c`、`Xj+Xk > c` 或分支均值精确抵消；
6. 将每个 DGP 的叶节点概率、根节点条件均值和理论/蒙特卡洛分裂收益写入诊断表。

如果检查失败，停止该 DGP，不通过临时改模型参数掩盖结构问题。

## 6. 配置矩阵

### 6.1 面板 A：标准树家族

```text
structure in {tree_stumps, tree_hierarchical, tree_forest_sum}
n in {1000, 2000}
p in {10, 50}
```

共 `3 * 2 * 2 = 12` 个配置。

### 6.2 面板 B：小样本高维压力

```text
structure in {tree_stumps, tree_hierarchical, tree_forest_sum}
n in {300, 500}
p in {50, 100}
```

共 `3 * 2 * 2 = 12` 个配置。函数公式和噪声与面板 A 完全一致；超过真实函数需要的维度全部是独立无关变量。

两个面板合计 24 个配置。所有方法在同一配置、同一 replication 内共享数据、外层折分和评价样本。

## 7. 学习器与公平比较

### 7.1 TabICLv2

- `tabiclv2_1`：主要 Tab 方法；
- `tabiclv2_8`：预先规定的集成规模敏感性方法；
- 使用 GPU，最多一个 GPU 工人；
- 不根据模拟真值调整模型或人工添加 DGP 特征。

### 7.2 XGBoost

- 现有默认 XGBoost 作为历史基准；
- 主要比较对象为每个配置分别冻结、且对 `l` 与 `m` 分别选择的 tuned-XGBoost；
- XGBoost 使用 CPU，最多八个 CPU 工人；硬件差异只影响运行时间，不影响数据、目标和指标。

候选网格固定为：

| candidate | n_estimators | max_depth | learning_rate | min_child_weight | reg_lambda | subsample | colsample_bytree |
|---|---:|---:|---:|---:|---:|---:|---:|
| xgb_d1_lr003 | 800 | 1 | 0.03 | 1 | 1 | 0.9 | 1.0 |
| xgb_d2_lr003 | 800 | 2 | 0.03 | 1 | 1 | 0.9 | 1.0 |
| xgb_d2_lr005 | 600 | 2 | 0.05 | 5 | 1 | 0.9 | 1.0 |
| xgb_d3_lr003 | 800 | 3 | 0.03 | 5 | 1 | 0.9 | 1.0 |
| xgb_d3_lr005 | 600 | 3 | 0.05 | 10 | 2 | 0.9 | 1.0 |
| xgb_d4_lr003 | 800 | 4 | 0.03 | 5 | 2 | 0.9 | 1.0 |

正式选择只能依据验证集上可观测的 `Y`-MSE 或 `D`-MSE。`l0` 和 `m0` 只用于模拟诊断，禁止参与正式模型选择。

### 7.3 补充方法

- ExtraTrees：作为另一种树模型基线；
- Oracle：只用于机制诊断；
- Oracle 结果不得参与“可实际使用的方法”排名。

## 8. Stage 4A：探索与冻结

Stage 4A 分为两个不共享随机种子的步骤。

### 8.1 Stage 4A-Tune：XGBoost 参数筛选

- 每个配置 10 个 tuning replication；
- 每次使用固定的 75% 训练集和 25% 验证集；
- `l` 候选用验证 `Y`-MSE 排名，`m` 候选用验证 `D`-MSE 排名；
- 在 10 次 replication 上分别求平均损失；
- 每个配置分别冻结一个 `l` 参数哈希和一个 `m` 参数哈希；
- 保存全部候选损失，不只保存优胜模型。

### 8.2 Stage 4A-Screen：DGP 筛选

- 使用新的 seed namespace；
- 每个配置 20 个 screening replication；
- 5 折交叉拟合；
- 比较 `tabiclv2_1/tabiclv2_1`、`tabiclv2_8/tabiclv2_8`、默认 XGBoost、冻结 tuned-XGBoost 和 ExtraTrees；
- 保存 Bias、RMSE、Empirical SD、Mean SE、Coverage、区间长度、`l_mse`、`m_mse`、`lm_error_cross`、运行时间和失败信息。

### 8.3 确认候选选择规则

主要 Tab 方法固定为 `tabiclv2_1`，主要传统比较对象固定为 tuned-XGBoost。对每个“面板 × 结构”单元：

1. 计算 TabICLv2-1 相对 tuned-XGBoost 的配对平方误差差值；
2. 在该单元的四个 `n,p` 配置中，选择平均配对平方误差差值最小的配置；
3. 即使四个配置中 TabICLv2 全部落后，也选择差值最小者并据实进入确认；
4. 不以 `m0/l0` 真值预测误差选择确认配置；这些指标只用于解释；
5. 冻结六个确认配置：面板 A 三个结构各一个，面板 B 三个结构各一个。

该规则保证面板 B 不会因为结果不利而被删除，也避免只确认一个偶然获胜的配置。

## 9. Stage 4B：独立确认

- 六个冻结配置；
- 每个配置先运行 5 个 smoke replication；
- smoke 通过后使用全新 seed namespace 补齐到 100 个正式 replication；
- XGBoost 参数、DGP 公式、选择规则和分析代码在首个正式 replication 前冻结；
- 每个 learner/target/replication 的 OOF prediction 只训练一次并缓存；
- 所有 DML 组合复用相同缓存，避免重复计算和随机差异。

正式方法包括：

1. TabICLv2-1 / TabICLv2-1；
2. TabICLv2-8 / TabICLv2-8；
3. 默认 XGBoost / 默认 XGBoost；
4. tuned-XGBoost-l / tuned-XGBoost-m；
5. ExtraTrees / ExtraTrees；
6. Oracle / Oracle。

每个配置另做 Oracle-l/learned-m 与 learned-l/Oracle 误差来源诊断，但不纳入主要优越性检验。

## 10. 统计分析和结论规则

### 10.1 主要比较

主要比较是每个确认配置中：

```text
TabICLv2-1 / TabICLv2-1
vs
tuned-XGBoost-l / tuned-XGBoost-m
```

主要终点为同一 replication 下的处理效应平方误差差：

```text
(theta_hat_tab - theta0)^2 - (theta_hat_xgb - theta0)^2
```

对六个确认配置分别进行双侧配对检验，并使用 Holm 方法控制六次主要比较的家族错误率。

### 10.2 单配置优越性声明

只有同时满足以下条件，才声明 TabICLv2 在该树状 DGP 配置下优于 XGBoost：

1. TabICLv2 的 RMSE 至少低 10%；
2. 配对检验 Holm 校正后 `p < 0.05`；
3. TabICLv2 的 Coverage 不得比 tuned-XGBoost 低超过 0.05；
4. TabICLv2 的 Coverage 必须至少为 0.90；
5. 没有失败、OOM、静默 fallback 或结果缺失造成的不对称样本。

若 nuisance MSE 更低但 DML RMSE 未显著改善，只能报告“改善了 nuisance prediction，未转化为显著的处理效应估计优势”。

### 10.3 面板层面的解释

- 若面板 A 至少两个结构满足优越性条件，可表述为“优势跨越多个标准树结构”；
- 若只有面板 B 满足，可表述为“优势主要集中于小样本高维树状环境”；
- 若只有一个配置满足，只报告配置特异性结果，不推广为一般树状结论；
- 若没有配置满足，保留完整负面结果，并将平滑 DGP 优势与树状 DGP 负面结果共同构成适用边界。

### 10.4 次要分析

- Bias、绝对偏差、Empirical SD、Mean SE 和 95% Coverage；
- `l_mse`、`m_mse`、`lm_error_cross` 和理论偏差代理；
- TabICLv2-8 的敏感性结果；
- 相同配置随 `n`、`p` 变化的趋势；
- 运行时间、CPU 工时和 GPU 峰值显存；
- Coverage 使用精确二项区间，并检验名义值 0.95 是否落在区间内。

## 11. 运行、断点续跑与产物

建议使用独立命名空间和目录：

```text
configs/stage4_tree_benchmark.yaml
results/stage4_tree_structure_checks/
results/stage4_tree_tuning_raw/
results/stage4_tree_tuning/
results/stage4_tree_screening_raw/
results/stage4_tree_screening/
results/stage4_tree_confirmation_raw/
results/stage4_tree_confirmation/
results/stage4_tree_cache/
results/logs/stage4_tree/
```

要求：

- 任务键包含阶段、面板、结构、`n`、`p`、replication、target、learner 和配置哈希；
- 数据种子、折分种子和模型种子由稳定命名空间派生；
- JSON 和缓存采用原子写入；
- 已成功任务可跳过，失败任务可单独重试；
- GPU 队列最多一个 TabICLv2 工人，CPU 队列最多八个工人；
- 进度文件记录计划数、成功数、失败数、运行中任务、开始时间和更新时间；
- 关机后能够依据成功 JSON 和缓存继续，而不是从头运行；
- 正式结果发布到新的 `results/published/stage4_tree_benchmark/`，不覆盖历史结果。

## 12. 测试与验收

### 12.1 代码测试

- 三个新场景在固定 seed 下确定性复现；
- `tree_stumps` 与现有 `tree_simple` 在相同输入下结构值一致；
- S2 的四个叶值和分支路径正确；
- S3 等于两棵指定层级树之和；
- 结构验收能够拒绝已知 XOR 例子；
- 配置枚举恰好得到面板 A 12 个、面板 B 12 个配置；
- tuning 不读取 `m0/l0` 作为选择损失；
- 每个配置冻结独立的 `l` 和 `m` XGBoost 哈希；
- confirmation 选择器每个“面板 × 结构”恰好选择一个配置；
- 缓存、分片、断点续跑、损坏检测和结果聚合均有测试；
- 旧 DGP 和旧实验测试保持通过。

### 12.2 Smoke 验收

- 结果数量与任务枚举一致；
- 所有数值有限、置信区间有序；
- Oracle/Oracle 在 Monte Carlo 误差范围内无偏；
- 同一 replication 的所有方法具有相同 data seed 和 fold seed；
- XGBoost 与 TabICLv2 均无未记录 fallback；
- 并行运行不会让两个 TabICLv2 进程同时占用 GPU。

### 12.3 研究验收

- 24 个筛选配置全部报告；
- 六个确认配置严格按预设规则生成；
- 100 次独立确认完整结束；
- 主要比较、Holm 校正、Coverage 约束和 nuisance 解释齐全；
- 无论 TabICLv2 是否胜出，都生成中文分析报告、机器可读 CSV、图形和复现实验命令。

## 13. 论文中的预期定位

Stage 4 最终用于界定 TabICLv2-DML 的适用边界，而不是制造单方面胜利：

- 原复杂 `tree`：纯交互和斜边界压力测试；
- 标准树面板：树专用模型的常规优势或 TabICLv2 的跨结构竞争力；
- 小样本高维面板：TabICLv2 是否因预训练归纳偏置在数据稀缺时获得相对优势；
- 现有 `smooth`：TabICLv2 已表现出稳定优势的平滑非线性参照组。

只有独立确认结果满足第 10 节规则时，论文才使用“TabICLv2 优于 XGBoost”的表述；否则使用更窄、更符合证据的结论。
