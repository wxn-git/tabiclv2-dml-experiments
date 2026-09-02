# Stage 3B `tree_simple` 重跑设计

## 目标

新增一个比现有 `tree` 更容易学习的阈值型 DGP，并完整重跑 Stage 3B，用于检验原 tree 欠覆盖是否主要来自乘积交互和非轴对齐边界，而不是 TabICLv2 或 DML 实现错误。

## DGP

保留原 `tree` 不变，新增 `tree_simple`：

\[
m_{raw}(X)=0.9I(X_0>0)-0.7I(X_1>0)+0.5I(X_2>0)
\]

\[
g_{raw}(X)=0.8I(X_0>0)+0.6I(X_3>0)-0.5I(X_4>0)
\]

与原实验相同，`m_raw` 和 `g_raw` 分别使用现有 `_unit_scale` 标准化，然后生成

\[
D=m_0(X)+V,\qquad Y=\theta_0D+g_0(X)+\varepsilon.
\]

只改变结构函数，保持噪声、标准化和估计流程不变，以便把新旧差异归因于结构简化。所有阈值均为单变量、轴对齐阈值，浅层树无需用多个矩形区域近似乘积或斜线边界。

## 实验范围

固定 `n=2000`、`p=10`、`theta0=1`、5 折交叉拟合。

- Batch A：Oracle、TabICLv2-1、XGBoost 的 9 个 `l/m` 组合，各 50 次，共 450 条 DML 结果。
- Batch B：原 Stage 3B 的 17 个处理学习器候选，各 10 次，共 170 条筛选结果。
- Batch C：3 个 `l` 候选与 5 个 `m` 候选的 15 个组合，各 50 次，共 750 条独立确认结果。

Batch B 仍只按可观察的验证集 `D`-MSE 选择每个候选家族的优胜者；真实 `m0`-MSE 只作诊断，不能参与选择。

## 隔离与可复现性

- 使用场景名 `tree_simple`，不修改 `tree`。
- 使用新的 stage 和 seed namespace：`stage3b_tree_simple_batch_a`、`stage3b_tree_simple_screening`、`stage3b_tree_simple_confirmation`。
- 所有缓存、原始结果、日志、入选配置和分析文件写入带 `tree_simple` 的新目录。
- 不读取或覆盖原 Stage 3B 结果；新旧比较只在聚合阶段进行。
- TabICLv2 使用 GPU；XGBoost、ExtraTrees 和 Oracle 使用 8 个 CPU 工人，与原实验一致。

## 验证顺序

1. 先写 DGP 行为测试并确认在实现前失败。
2. 实现 `tree_simple` 后运行目标测试和全套测试。
3. 三批各执行最小 smoke test，检查任务数、缓存隔离、成功状态和聚合。
4. smoke 全部通过后启动正式 Batch A；完成后运行 Batch B，并冻结胜出配置；最后运行 Batch C。
5. 正式结果完成后检查预期文件数 `450/170/750` 和全部 `status=success`。

## 主要评价指标与判定

报告 Bias、RMSE、95% Coverage、平均标准误、经验标准差、区间长度、`l`-MSE、`m`-MSE、误差交叉项和代理偏差。

核心比较为：

1. `tree_simple` 是否显著降低可行处理学习器的 `m`-MSE；
2. Oracle-`l` + learned-`m` 的衰减偏差是否减小；
3. XGBoost/XGBoost、TabICLv2/TabICLv2 和混合组合的覆盖率是否恢复；
4. 若结构变简单后结果恢复，则原 tree 的困难主要来自复杂决策边界；若仍不恢复，则继续考察带噪处理回归、样本量和 DML 有限样本条件。

本次重跑是机制对照，不预设“必须改善”的结论。
