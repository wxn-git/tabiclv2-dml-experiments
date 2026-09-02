# Stage 3B `tree_simple` 结果解释

## 完整性

- Batch A：450/450 成功；
- Batch B：170/170 成功；
- Batch C：750/750 成功；
- 所有正式记录均为 `tree_simple`，错误日志为空。

## 新旧 tree 核心对照

| `l` / `m` | 原tree Bias | simple Bias | 原tree RMSE | simple RMSE | 原tree Coverage | simple Coverage |
|---|---:|---:|---:|---:|---:|---:|
| TabICLv2 / TabICLv2 | -0.0928 | -0.0022 | 0.0958 | 0.0248 | 0.06 | 0.88 |
| XGBoost / XGBoost | -0.0494 | -0.0115 | 0.0550 | 0.0256 | 0.46 | 0.94 |
| TabICLv2 / XGBoost | -0.0612 | -0.0334 | 0.0660 | 0.0408 | 0.26 | 0.70 |
| XGBoost / TabICLv2 | -0.1132 | -0.0154 | 0.1154 | 0.0286 | 0.00 | 0.90 |
| XGBoost / tuned XGBoost | -0.0573 | -0.0091 | 0.0628 | 0.0250 | 0.32 | 0.92 |

简化为单变量轴对齐阈值后，TabICLv2 的 `m`-MSE 从 0.1817 降到 0.0469，XGBoost 的 `m`-MSE 从 0.1274 降到 0.0748。结果支持：原复杂 tree 的乘积交互和非轴对齐边界是辅助学习困难的重要来源；当前证据不支持存在一个固定的 DML 实现错误。

## 理论警告

Oracle-`l` + TabICLv2-`m` 的 Coverage 仍只有 0.34，说明处理模型误差单独存在时仍会造成衰减。TabICLv2/TabICLv2 的 Coverage 恢复到 0.88，还因为：

\[
m\text{-MSE}=0.0469,
\qquad E[\delta_l\delta_m]=0.0508,
\]

二者在偏差公式中几乎抵消。当前 `m0` 与 `g0` 共用 `I(X0>0)`，可能增强这种有利误差相关。因此不能把该结果直接推广为“简单 tree 下 TabICLv2-DML 总能有效推断”。

论文中应同时保留原 `tree` 和 `tree_simple`，分别作为复杂阈值压力测试与简单轴对齐阈值场景。下一项控制实验应让简单 `m0` 和 `g0` 使用不重叠特征，以区分函数可学习性与误差抵消。
