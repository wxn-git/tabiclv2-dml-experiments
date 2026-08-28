# Stage 3B 面向论文发表的 Tree 机制诊断与处理模型改进设计

## 1. 研究目标

本阶段不删除或弱化现有 tree DGP，也不通过扩大标准误掩盖偏差。目标是形成可用于论文的证据链：

1. 定量验证处理干扰函数 `m(X)=E[D|X]` 的误差如何产生 DML 负偏差；
2. 在不使用 Oracle 信息训练正式模型的前提下，筛选更适合 tree 结构的 `m` 学习器；
3. 检验降低 `m` 的预测误差后，处理效应 Bias、RMSE 和 Coverage 是否同步改善；
4. 为后续使用全新种子、200 至 500 次重复的确认性实验冻结候选方法。

当前阶段属于机制诊断和方法筛选，不作为论文最终覆盖率表。

## 2. 研究假设

- H1：现有 tree 欠覆盖主要由 `m` 的平方误差引起，而不是 DGP、DML 公式或 GPU/CPU 差异引起。
- H2：`l` 与 `m` 的误差相关项可以部分抵消或放大 `m` 平方误差造成的偏差，因此只报告两个 MSE 的乘积不充分。
- H3：tree 原生学习器或经过独立筛选的 XGBoost 配置可以降低 `m_mse`。
- H4：当 `m_mse` 下降时，DML 的绝对偏差和 RMSE 应下降，Coverage 应提高。

## 3. 固定数据设计

- DGP：`tree`；
- 真值：`theta0=1.0`；
- 基准规模：`n=2000, p=10`；
- 外层交叉拟合：5 folds；
- TabICLv2：GPU；
- XGBoost、ExtraTrees、Oracle 和调度：CPU；
- 最多一个 TabICLv2 GPU 工人和八个 CPU 工人；
- 所有候选方法在相同 replication 内共享数据与折分；
- Stage 1、Stage 2 和 Stage 3A 原始结果保持只读，不覆盖。

## 4. Batch A：现有结果的精确误差分解

### 4.1 数据和重复

- 使用 Stage 3A 相同的 `seed_namespace=stage3_tree_diagnosis`；
- 复算 replications `0..49`；
- 学习器为 `tabiclv2_1`、当前 XGBoost 和 Oracle；
- 每个 replication 的每个 nuisance-target 预测只训练一次并缓存，所有 `l/m` 组合复用相同 OOF 预测。

### 4.2 新增诊断量

令：

```text
delta_l = l_hat - l0
delta_m = m_hat - m0
```

每个任务保存：

- `l_mse = mean(delta_l^2)`；
- `m_mse = mean(delta_m^2)`；
- `lm_error_cross = mean(delta_l * delta_m)`；
- `residual_d_variance = mean((D-m_hat)^2)`；
- `bias_numerator_proxy = lm_error_cross - theta0*m_mse`；
- `theta_proxy = (theta0 + lm_error_cross) / (1 + m_mse)`，仅用于当前 `Var(V)=1` 的机制解释；
- `proxy_error = theta_hat - theta_proxy`。

Oracle/Oracle 必须继续与直接 DML 计算完全一致。Oracle/learned-m 的平均结果应与 `1/(1+m_mse)` 的理论衰减方向一致；这是机制检验，不设定为了通过而必须达到的数值阈值。

## 5. Batch B：处理模型 m 的独立筛选

### 5.1 筛选数据

- 新种子空间：`stage3b_mscreen_pilot`；
- 10 个独立 replication；
- 每个 replication 使用固定的训练/验证拆分；
- 模型选择依据只能是验证集上可观测的 `D` 预测 MSE；
- `m0` 只用于模拟诊断和事后解释，禁止用于正式候选模型的选择或训练。

### 5.2 候选模型

1. TabICLv2-1；
2. TabICLv2-8；
3. 当前 XGBoost 基准；
4. 小型 XGBoost 候选网格，覆盖树深、学习率、叶节点最小样本约束和正则化；
5. ExtraTrees 小型候选网格，覆盖 `max_features` 和 `min_samples_leaf`。

XGBoost 和 ExtraTrees 的最优配置按 10 个 replication 的平均验证 `D` MSE 选择并冻结。筛选结果只用于决定 Batch C 候选，不作为正式性能结论。

另运行一个标记为 `oracle_target_diagnostic` 的小型诊断：让 TabICLv2-1 和当前 XGBoost 学习无噪声 `m0`。该结果只用于区分“结构近似困难”和“带噪声目标学习困难”，不得与公平基线合并或作为可部署方法。

## 6. Batch C：改进后的 DML 独立确认

### 6.1 数据和方法

- 新种子空间：`stage3b_confirmation`；
- 先运行 5 个 replication 的 smoke；
- smoke 通过后自动补齐到 50 个 replication；
- `l` 候选：TabICLv2-1、当前 XGBoost、Oracle；
- `m` 候选：TabICLv2-1、当前 XGBoost、冻结的 tuned-XGBoost、冻结的 ExtraTrees、Oracle；
- 采用 nuisance prediction cache，单个 learner/target/replication 只拟合一次。

### 6.2 主要指标

- Bias、RMSE、Empirical SD、Mean SE；
- 95% Coverage 和区间长度；
- `l_mse`、`m_mse`、`lm_error_cross`；
- 理论偏差代理与实际偏差的差异；
- 单次运行时间、总运行时间、TabICLv2 峰值显存；
- 相同 replication 下的配对差异与 Monte Carlo 标准误。

### 6.3 科学判断规则

- 不以单个 replication 或单一 Coverage 数字判断优劣；
- 重点检验 `m_mse` 排名、DML RMSE 排名和 Coverage 改善方向是否一致；
- 如果 tuned-XGBoost 或 ExtraTrees 没有改善，保留负面结果，不临时修改 DGP；
- 如果改善明显，将优胜 `m` 学习器带入后续样本量实验；
- 如果所有可行学习器仍严重欠覆盖，将当前 tree DGP明确报告为有限样本压力测试。

## 7. 实现与可复现性要求

- 新增结果目录，不覆盖既有 JSON；
- 每个任务使用确定性 task key、data seed、fold seed 和 learner seed；
- 缓存采用原子写入，成功结果可断点续跑；
- 缓存内容包含预测数组、真值、数据/折分种子、模型名和配置哈希；
- 模型配置、依赖版本、GPU信息和运行命令写入环境记录；
- CPU 与 GPU 的差异只影响运行速度，不改变数据、损失函数或评价指标；
- 任何失败、OOM 或 fallback 单独记录，禁止静默替换模型；
- 先写测试，再实现缓存、误差分解、模型筛选和聚合。

## 8. 验收标准

### 实现验收

- 测试覆盖缓存复用、缓存损坏检测、异构 `l/m` 组合、Oracle、误差分解公式、任务键、分片和断点续跑；
- smoke 无重复任务、无缺失任务、数值有限且置信区间有序；
- 同一 seed 下旧 Stage 3A 学习器的 OOF 预测和处理效应结果保持兼容；
- 结果数量与配置枚举完全一致；
- stderr 无未解释错误。

### 研究验收

- Batch A 能报告理论偏差代理与实际估计的配对关系；
- Batch B 冻结 tuned-XGBoost 和 ExtraTrees 配置，不在 Batch C 中继续调参；
- Batch C 给出 50 次独立重复的完整指标和配对比较；
- 无论假设是否得到支持，都保留完整结果并据实报告。

## 9. 本阶段之后

Stage 3B 完成后再决定是否进入论文确认性实验：

- `n in {500, 1000, 2000, 5000, 10000}` 的收敛曲线；
- 预先锁定配置和全新种子；
- 主要表格使用至少 200 次、算力允许时 500 次重复；
- 增加真实数据案例；
- 在文献检索后决定将贡献表述为系统实证研究还是加入正式的 Hybrid DML 方法。
