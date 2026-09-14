# 实验结果与结论边界

## 1. 评价指标

所有模拟设定真实处理效应 `theta0 = 1`。主要指标包括：

- **Bias（偏差）**：平均估计值减去真实值。越接近 0 越好。
- **RMSE（均方根误差）**：同时惩罚偏差和随机波动。越小越好。
- **Coverage（覆盖率）**：95% 置信区间包含真实值的比例。理想情况下接近 0.95。
- **`l_mse`**：结果 nuisance function `l(X)=E[Y|X]` 的预测误差。
- **`m_mse`**：处理 nuisance function `m(X)=E[D|X]` 的预测误差。
- **Runtime**：当前机器和当前并行实现下的运行时间。

RMSE 低说明点估计准确，但不能自动保证置信区间可靠。Coverage 很低通常意味着剩余偏差相对于标准误过大。

## 2. Stage 1：筛选而非最终结论

Stage 1 使用 20 次重复扫描 48 个数据配置。它的作用是为 Stage 2 选择代表性场景，不足以提供稳定的覆盖率结论。

按场景汇总，TabICLv2 在 mixed 的 12 个配置中赢得 9 个，在 smooth 中赢得 7 个，在线性中赢得 5 个，在 tree 中没有赢得任何配置。tree 场景的平均相对改进为负，提示必须保留负面场景进入正式实验。

Stage 2 最终选出：

- 3 个 smooth 配置作为优势端；
- 3 个 tree 配置作为劣势端；
- 1 个 linear 配置作为简单基准。

数据来源：[`stage1_scenario_summary.csv`](results/published/stage1/stage1_scenario_summary.csv) 和 [`stage1_selected_configs.csv`](results/published/stage1/stage1_selected_configs.csv)。

## 3. Stage 2：100 次正式重复

Stage 2 比较 Lasso、随机森林、XGBoost、MLP、传统 ensemble、TabICLv2-1 和 TabICLv2-8。每个“配置 × learner”组合有 100 次蒙特卡洛重复；同一次重复共享数据种子和交叉拟合种子，因此可以做配对比较。

### 3.1 平滑非线性：TabICLv2 的主要正面证据

| 配置 | 最佳传统模型 RMSE | 最佳 Tab RMSE | 相对改进 | Tab 覆盖率 |
| --- | ---: | ---: | ---: | ---: |
| smooth, n=500, p=50 | 0.0578 | 0.0497 | 约 14.1% | 0.94 |
| smooth, n=1000, p=50 | 0.0378 | 0.0351 | 约 7.0% | 0.95 |
| smooth, n=1000, p=100 | 0.0431 | 0.0379 | 约 11.9% | 0.91 |

三个配置中的最佳传统模型均为 ensemble。TabICLv2-1 每次约 4–5 秒，TabICLv2-8 约 9–15 秒，而 ensemble 在当前实现中约为 21–62 分钟。统计准确率可以直接比较；运行时间只能代表当前硬件和实现。

数据来源：[`tab_vs_best_traditional.csv`](results/published/stage2/tab_vs_best_traditional.csv)。

### 3.2 线性：精度持平，Lasso 更合理

在 `linear_n2000_p50` 中：

- ensemble RMSE 为 0.0248；
- Lasso RMSE 为 0.0250；
- TabICLv2-1 RMSE 为 0.0250；
- TabICLv2-8 RMSE 为 0.0250。

这些差异很小。TabICLv2 比当前 ensemble 快，但 Lasso 每次约 0.11 秒，结构更简单。因此在线性关系已知或近似成立时，没有足够证据要求用 TabICLv2 替换 Lasso。

### 3.3 树状阈值：点估计和区间推断均出现问题

| 配置 | 最佳传统模型 | 传统 RMSE | TabICLv2-1 RMSE | 最佳传统覆盖率 | Tab 覆盖率 |
| --- | --- | ---: | ---: | ---: | ---: |
| tree, n=2000, p=10 | XGBoost | 0.0566 | 0.1000 | 0.42 | 0.01 |
| tree, n=5000, p=10 | XGBoost | 0.0438 | 0.0936 | 0.25 | 0.00 |
| tree, n=5000, p=50 | Lasso | 0.0712 | 0.1068 | 0.00 | 0.00 |

树状场景不是“只有 TabICLv2 表现差”。XGBoost 的点估计更准，但所有 learner 的覆盖率都远低于 0.95。这表明 DML 标准误没有吸收仍然存在的 nuisance estimation bias，不能只根据 RMSE 选择模型后就宣称推断有效。

完整覆盖率诊断见 [`coverage_diagnostics.csv`](results/published/stage2/coverage_diagnostics.csv)。

### 3.4 TabICLv2-1 与 TabICLv2-8

增加到 8 个估计器没有稳定提高 RMSE，运行时间通常增加约 2–4 倍。当前证据支持把 TabICLv2-1 作为主模型，把 TabICLv2-8 作为稳健性或消融设置，而不是默认方案。

Stage 2 的完整中文分析见 [`analysis_report_zh.md`](results/published/stage2/analysis_report_zh.md)。

## 4. Stage 3：定位树状设计中的失败通道

Stage 3 固定 tree、`n=2000`、`p=10`，分别让 `l` 和 `m` 使用 oracle、TabICLv2-1 或 XGBoost。

“Oracle”表示直接使用模拟中已知的真实 nuisance function。现实数据没有 oracle，但模拟实验可以用它做故障隔离：

- 固定真实 `m`，只让 `l` 有预测误差，可以检验结果模型是否是主要问题；
- 固定真实 `l`，只让 `m` 有预测误差，可以检验处理模型是否是主要问题。

Stage 3 发现：只要 `m` 使用 oracle，TabICLv2 或 XGBoost 的 `l` 都能产生接近无偏的处理效应；只要 `m` 改为学习值，明显的向下偏差就重新出现。该结论随后由 Stage 3B 使用独立种子确认。

## 5. Stage 3B：误差分解、筛选和独立确认

Stage 3B 包含三个批次：

- Batch A：450 个结果，复现 Stage 3 并加入误差分解；
- Batch B：170 个结果，使用 10 次独立筛选重复比较处理模型；
- Batch C：750 个结果，15 个 `l`/`m` 组合各做 50 次独立确认。

### 5.1 Batch A：`m` 误差能够解释主要偏差

当 `l` 为 oracle 时：

| `m` learner | `m_mse` | 理论代理 `theta_proxy` | 实际平均估计约值 |
| --- | ---: | ---: | ---: |
| oracle | 0.0000 | 1.0000 | 0.9982 |
| XGBoost | 0.1264 | 0.8878 | 0.8853 |
| TabICLv2-1 | 0.1810 | 0.8468 | 0.8454 |

代理值和实际估计非常接近，说明树状 DGP 的主要偏差可以由 treatment nuisance error 解释，而不是由标准误代码或 `l` 单独造成。

数据来源：[`batch_a_summary.csv`](results/published/stage3b/batch_a_summary.csv)。

### 5.2 Batch B：常规调参没有解决问题

候选选择只使用 validation `D` MSE：

| 候选 | validation `D` MSE | diagnostic `m0` MSE |
| --- | ---: | ---: |
| 当前 XGBoost | 1.1298 | 0.1405 |
| 最佳调参 XGBoost | 1.1335 | 0.1468 |
| TabICLv2-8 | 1.1721 | 0.1949 |
| TabICLv2-1 | 1.1724 | 0.1932 |
| 最佳 ExtraTrees | 1.1872 | 0.2078 |

当前 XGBoost 的 validation `D` MSE 最低；有限候选集中的调参 XGBoost 和 ExtraTrees 没有改进。直接用模拟真值 `m0` 训练的诊断实验中，当前 XGBoost 的 `m0` MSE 为 0.0194，TabICLv2-1 为 0.0919，但这种真值目标在现实数据中不可观测，不能用于实际模型选择。

数据来源：[`screening_summary.csv`](results/published/stage3b/screening_summary.csv)。

### 5.3 Batch C：独立确认

| `l` / `m` 组合 | Bias | RMSE | Coverage |
| --- | ---: | ---: | ---: |
| oracle / oracle | -0.0036 | 0.0199 | 1.00 |
| XGBoost / oracle | -0.0030 | 0.0239 | 0.98 |
| TabICLv2-1 / oracle | -0.0020 | 0.0249 | 0.92 |
| XGBoost / XGBoost | -0.0494 | 0.0550 | 0.46 |
| XGBoost / tuned XGBoost | -0.0573 | 0.0628 | 0.32 |
| TabICLv2-1 / XGBoost | -0.0612 | 0.0660 | 0.26 |
| TabICLv2-1 / TabICLv2-1 | -0.0928 | 0.0958 | 0.06 |

当 `m` 为 oracle 时，学习得到的 `l` 并不会造成明显偏差；当 `m` 使用当前可用 learner 时，偏差和欠覆盖重新出现。这是“`m(X)` 是主要瓶颈”的最直接证据。

数据来源：[`confirmation_summary.csv`](results/published/stage3b/confirmation_summary.csv) 和 [`analysis_report_zh.md`](results/published/stage3b/analysis_report_zh.md)。

## 6. 论文中可以怎样表述

当前证据支持：

> 在平滑高维非线性设计中，TabICLv2 作为 PLR-DML nuisance learner，相比当前最佳传统集成方法降低了处理效应 RMSE，并保持接近名义水平的覆盖率。该优势具有明显的 DGP 依赖性，不能推广到树状阈值设计。在树状设计中，主要失败通道是 treatment nuisance function `m(X)` 的估计误差；有限的 XGBoost 调参和 ExtraTrees 筛选没有消除该问题。

当前证据不支持：

- “TabICLv2 在所有 DGP 上都优于传统机器学习”；
- “树状设计的失败证明 TabICLv2 完全不会学习树结构”；
- “50 次 Stage 3B 重复足以给出最终精确覆盖率”；
- “合成数据结论能够直接推广到现实因果数据”；
- “CPU/GPU 运行时间差异代表算法的绝对速度差异”。

## 7. 下一步实验

Stage 4 已完成原计划中的树状冻结配置确认。若目标是论文投稿，下一步建议：

1. 保留 Stage 4 的 `0/6` 结果，不在看到结果后放宽优越性门槛；
2. 若要继续追踪 `tree_hierarchical` 信号，应新建探索性实验并使用新的 seed namespace，不能把追加重复并入现有确认性检验；
3. 增加真实或半合成数据，检验 smooth 优势是否具有外部意义；
4. 将 Stage 2 二次分析完全脚本化，并统一报告 Monte Carlo standard error；
5. 把 tree 结果作为适用边界和机制性发现，而不是删除不利场景。

## 8. `tree_simple` 轴对齐阈值重跑

为检验原 tree 结果是否主要来自复杂决策边界，新增仅含单变量轴对齐阈值的 `tree_simple`，其余样本量、噪声、交叉拟合和学习器配置不变。三批正式结果分别为 450、170 和 750 条，全部成功。

独立确认中，TabICLv2/TabICLv2 的 Bias、RMSE、Coverage 从 `-0.0928 / 0.0958 / 0.06` 改善为 `-0.0022 / 0.0248 / 0.88`；XGBoost/XGBoost 从 `-0.0494 / 0.0550 / 0.46` 改善为 `-0.0115 / 0.0256 / 0.94`。TabICLv2 的处理模型 MSE 从 0.1817 降到 0.0469。

该结果说明复杂阈值边界是原实验失败的重要来源，但改善也包含 `l/m` 误差交叉项的有利抵消。完整汇总与新旧对照见 [`results/published/stage3b_tree_simple/`](results/published/stage3b_tree_simple/)。

## 9. Stage 4：轴对齐树状 DGP 确认性基准

### 9.1 为什么还需要 Stage 4

Stage 3B 的 `tree_simple` 说明：把原始树状 DGP 简化为轴对齐阈值后，TabICLv2 和 XGBoost 都能明显恢复。但单个简化 DGP 仍不能回答“TabICLv2 是否在树状结构下稳定优于 XGBoost”。Stage 4 因此扩大树结构和样本面板，并在查看最终确认结果前冻结选择规则。

实验包含三个树结构：

- `tree_stumps`：若干单变量阈值项相加，边界最简单；
- `tree_hierarchical`：上层阈值决定下层分支，包含层级交互；
- `tree_forest_sum`：多个浅树函数相加，模拟较丰富但仍轴对齐的树集成结构。

每种结构在 `standard` 和 `small_n_high_p` 两个面板中筛选，共 24 个候选配置。XGBoost 的 `l` 和 `m` 超参数分别通过独立 tuning 冻结，再从每个 `panel × structure` 组冻结一个确认配置。确认阶段使用新的随机种子，每个配置运行 100 次重复。最终发布产物包含 2,880 条 tuning 记录、4,800 条 screening 记录和 6,000 条 confirmation DML 记录，完整性检查未发现失败、OOM、缺失或静默 fallback。

### 9.2 预声明优越性规则

只有一个冻结配置同时满足以下五项，才声明 TabICLv2-1 优于 tuned-XGBoost：

1. 处理效应 RMSE 改善至少 10%；
2. 六项主要比较的 Holm 校正 `p` 值严格小于 0.05；
3. TabICLv2 coverage 不得比 tuned-XGBoost 低超过 0.05；
4. TabICLv2 coverage 至少为 0.90；
5. 所有预定方法组合按相同 replication 完整配对，且无失败、OOM、缺失或 fallback。

这些条件同时约束点估计精度、重复检验、区间推断和运行完整性，不能只选择其中对 TabICLv2 有利的指标。

### 9.3 六项冻结比较

| 面板 | 树结构 | `n` | `p` | Tab RMSE | tuned-XGB RMSE | RMSE 改善 | Holm `p` | Tab / XGB coverage | 全部通过 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| small_n_high_p | tree_forest_sum | 500 | 50 | 0.0498 | 0.0557 | 10.57% | 0.6370 | 0.95 / 0.92 | 否 |
| small_n_high_p | tree_hierarchical | 300 | 50 | 0.0674 | 0.0733 | 8.03% | 0.6370 | 0.90 / 0.88 | 否 |
| small_n_high_p | tree_stumps | 300 | 50 | 0.0646 | 0.0734 | 12.02% | 0.2721 | 0.94 / 0.89 | 否 |
| standard | tree_forest_sum | 1000 | 50 | 0.0324 | 0.0341 | 5.07% | 0.9531 | 0.92 / 0.93 | 否 |
| standard | tree_hierarchical | 1000 | 50 | 0.0370 | 0.0458 | 19.25% | 0.0501 | 0.89 / 0.83 | 否 |
| standard | tree_stumps | 1000 | 10 | 0.0379 | 0.0372 | -1.94% | 0.9531 | 0.91 / 0.89 | 否 |

TabICLv2-1 在 5/6 个配置中取得更低 RMSE，3/6 达到至少 10% 的改善幅度，但符合全部五项条件的配置数为 **0/6**。因此这些结果可以称为“方向性改善”，不能称为“统计确认的优越性”。

最强信号来自 `standard / tree_hierarchical`：TabICLv2-1 的 RMSE 低 19.25%，未校正配对 `p=0.0083`，平方误差差的 95% 置信区间也完全低于 0。但六项比较经 Holm 校正后 `p=0.0501`，略高于预声明的严格阈值；同时 TabICLv2 coverage 为 0.89，低于 0.90 门槛。事后把 0.0501 解释成“显著”或把 0.89 四舍五入成“达到 0.90”，都会破坏确认性设计。

### 9.4 Nuisance 预测与 DML 结果的关系

在 6 个冻结配置中，有 4 个配置的 `l_mse` 和 `m_mse` 都低于 tuned-XGBoost：两个 `tree_forest_sum` 和两个 `tree_hierarchical` 配置。可是 nuisance prediction 改善并没有稳定转化为满足门槛的处理效应优势。这与 DML 的正交化原理并不矛盾：DML 估计误差受 `l`、`m` 的误差大小、二者的相关方向以及有限样本波动共同影响，单独降低两个平均预测 MSE 并不保证处理效应 RMSE 按固定比例下降，也不保证置信区间覆盖率改善。

两个 `tree_stumps` 配置没有出现一致的 nuisance 优势。尤其在 `standard / tree_stumps` 中，TabICLv2-1 的 `l`、`m` 预测误差和处理效应 RMSE 均略差于 tuned-XGBoost。这一预先保留的落败配置可以防止只报告有利结构。

TabICLv2-8 在六项配置上的平均处理效应 RMSE 没有优于 TabICLv2-1，平均运行时间约为后者的 2.28 倍。因此 Stage 4 继续支持“TabICLv2-1 作为主设置、TabICLv2-8 作为稳健性设置”的选择。

### 9.5 论文结论边界

Stage 4 支持以下表述：

> 在预先冻结的六个轴对齐树状 DGP 配置中，TabICLv2-1 相对 tuned-XGBoost 在五个配置上呈现更低的处理效应 RMSE，并在部分层级树和浅树求和结构中同时改善了两个 nuisance function 的预测误差。然而，没有配置同时满足预声明的效应量、Holm 多重检验、覆盖率和完整性标准，因此本实验未获得 TabICLv2 在树状 DGP 上具有确认性优越性的证据。

这不是“TabICLv2 在树状数据上无效”，也不是“已经证明 TabICLv2 优于 XGBoost”。更准确的论文方向是：TabICLv2 作为 DML nuisance learner 的收益具有明显的 DGP 依赖性；平滑非线性场景已有较稳定正面结果，复杂树状场景暴露了 `m(X)` 瓶颈，而规范轴对齐树状场景呈现值得后续独立研究、但尚未通过确认门槛的信号。

正式报告见 [`analysis_report_zh.md`](results/published/stage4_tree_benchmark/analysis_report_zh.md)，六项配对统计见 [`primary_paired_comparisons.csv`](results/published/stage4_tree_benchmark/primary_paired_comparisons.csv)，完整冻结发布包见 [`results/published/stage4_tree_benchmark/`](results/published/stage4_tree_benchmark/)。
