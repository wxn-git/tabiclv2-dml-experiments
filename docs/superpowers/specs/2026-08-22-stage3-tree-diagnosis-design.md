# Stage 3 Tree 欠覆盖诊断与混合 DML 设计

## 目标

在不改变 Stage 1/2 原始结果和传统学习器实现的前提下，定位 tree 场景欠覆盖来自结果干扰函数 `l(X)`、处理干扰函数 `m(X)`，还是两者误差的联合效应，并检验 `TabICLv2/XGBoost` 异构学习器能否降低处理效应偏差。

## 已确认事实

- 自定义 PLR-DML 点估计和标准误在相同预测输入下与 DoubleML 一致。
- 三个 tree 配置的 1,000 次 Oracle 覆盖率分别为 0.938、0.949、0.955。
- 实际学习器的平均标准误与经验标准差大体接近，严重欠覆盖主要由持续负偏差造成。
- TabICLv2 继续使用 GPU；XGBoost 保持当前 CPU 实现，不更换为 GPU 版本。
- Stage 3A 使用 `tree, n=2000, p=10, folds=5, replications=50, tabicl_estimators=1`。

## 架构

### 异构交叉拟合

新增一个成对交叉拟合入口，分别接收 `learner_l_name` 与 `learner_m_name`。旧的 `crossfit_nuisances` 保持原签名，并委托给新入口，从而保证 Stage 1/2 不受影响。

`oracle` 是特殊学习器名称：`l` 侧直接使用 `data.l0[test]`，`m` 侧直接使用 `data.m0[test]`，不训练模型。

### 可配对随机种子

Stage 3 任务记录 `seed_namespace`。诊断批次使用 `stage3_tree_diagnosis`；正式确认批次使用 `stage2`，从而复用 Stage 2 的数据种子和折分种子。

每一侧模型的随机种子由该学习器对应的旧式 `TaskSpec` 键导出。这样 `Tab/Tab` 与旧 Tab 任务、`XGB/XGB` 与旧 XGBoost 任务可以逐值兼容；混合任务中的 Tab 侧和 XGBoost 侧也分别使用对应基线的模型种子。

### Stage 3 任务与存储

新增 `Stage3TaskSpec`，任务键包含：

```text
stage, scenario, n, p, replication, learner_l, learner_m, tabicl_estimators
```

Stage 3 写入独立目录 `results/stage3_tree_diagnosis_raw`，不覆盖 `results/raw`。

每条记录至少保存：`theta`、`standard_error`、置信区间、`l_mse`、`m_mse`、干扰误差乘积、两侧学习器、数据/折分种子、运行时间、显存、失败信息。

### 运行资源

含 TabICLv2 的五种组合由一个 GPU 子进程串行运行：

```text
tabiclv2_1/tabiclv2_1
tabiclv2_1/xgboost
xgboost/tabiclv2_1
oracle/tabiclv2_1
tabiclv2_1/oracle
```

不含 TabICLv2 的四种组合按 replication 稳定分片到最多八个 CPU 子进程：

```text
xgboost/xgboost
oracle/xgboost
xgboost/oracle
oracle/oracle
```

XGBoost 保持 `n_jobs=1`。GPU 子进程只有一个，避免争抢 8GB 显存。

## Stage 3A 方法组合

1. `tabiclv2_1/tabiclv2_1`
2. `tabiclv2_1/xgboost`
3. `xgboost/tabiclv2_1`
4. `xgboost/xgboost`
5. `oracle/tabiclv2_1`
6. `tabiclv2_1/oracle`
7. `oracle/xgboost`
8. `xgboost/oracle`
9. `oracle/oracle`

先用 `--replications 5` 运行 45 条冒烟任务。这 45 条属于最终 450 条的一部分；验证通过后补齐 replication 5-49。

## 验证与停止条件

- 单元测试先失败后通过，覆盖异构学习器、单边 Oracle、旧接口兼容、任务键和断点续跑。
- 新 `Tab/Tab` 与旧接口在固定种子下逐值一致；`XGB/XGB` 同样一致。
- 45 条冒烟任务必须全部写出，状态均为 `success`，无重复键、NaN、无穷值和非法标准误。
- Oracle/Oracle 点估计与直接调用 `estimate_plr_dml(data.y, data.d, data.l0, data.m0)` 一致。
- 任一 GPU OOM、种子不一致、旧接口不兼容或记录字段异常都会阻止自动补齐到 450 条。

## 非目标

- 本阶段不把 XGBoost、Lasso、随机森林或 sklearn MLP 改成 GPU 实现。
- 本阶段不调整 DML 标准误、不人为扩大置信区间。
- 本阶段不运行 TabICLv2-8。
- 本阶段不改动 Stage 1/2 原始结果。
- 分层 Tree-A/B/C、XGBoost 专项调参和重复交叉拟合在 Stage 3A/3B 结论后另行启动。

