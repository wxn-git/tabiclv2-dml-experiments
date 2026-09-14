# Stage 4 树状 DGP 确认性分析报告

## 预先固定的优越性规则

只有同一冻结配置同时满足以下五项条件，才声明 TabICLv2-1 优于 tuned-XGBoost：

1. RMSE 改善率至少为 10%（`>= 10%`）；
2. 六项主要比较的 Holm 校正 p 值严格小于 0.05；
3. TabICLv2 coverage 不得比 tuned-XGBoost 低超过 0.05；
4. TabICLv2 coverage 至少为 0.90；
5. 两种方法按相同 replication 完整配对，且十个预定方法组合没有失败、OOM、缺失或静默 fallback。

规则在查看确认结果前固定，本报告不作事后修改。

## 六项主要配对比较

| panel | structure | n | p | Tab RMSE | XGB RMSE | 改善率 | 均值平方误差差 | 差值 95% CI | 配对 p | Holm p | Tab 绝对误差胜率 | Tab coverage | XGB coverage | 优越 | 未满足条件 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| small_n_high_p | tree_forest_sum | 500 | 50 | 0.049812 | 0.055702 | 10.57% | -0.000621529 | [-0.00149108, 0.000248018] | 0.159254 | 0.637014 | 0.510 | 0.950 | 0.920 | 否 | holm_p_value_not_below_0.05 |
| small_n_high_p | tree_hierarchical | 300 | 50 | 0.067428 | 0.073313 | 8.03% | -0.000828327 | [-0.0020897, 0.000433045] | 0.195594 | 0.637014 | 0.510 | 0.900 | 0.880 | 否 | rmse_improvement_below_10pct;holm_p_value_not_below_0.05 |
| small_n_high_p | tree_stumps | 300 | 50 | 0.064595 | 0.073422 | 12.02% | -0.00121824 | [-0.00246007, 2.3602e-05] | 0.0544286 | 0.272143 | 0.620 | 0.940 | 0.890 | 否 | holm_p_value_not_below_0.05 |
| standard | tree_forest_sum | 1000 | 50 | 0.032356 | 0.034082 | 5.07% | -0.000114702 | [-0.000433203, 0.000203798] | 0.47655 | 0.9531 | 0.510 | 0.920 | 0.930 | 否 | rmse_improvement_below_10pct;holm_p_value_not_below_0.05 |
| standard | tree_hierarchical | 1000 | 50 | 0.036965 | 0.045780 | 19.25% | -0.000729388 | [-0.00126704, -0.000191739] | 0.00834415 | 0.0500649 | 0.630 | 0.890 | 0.830 | 否 | holm_p_value_not_below_0.05;tab_coverage_below_0.90 |
| standard | tree_stumps | 1000 | 10 | 0.037901 | 0.037181 | -1.94% | 5.4067e-05 | [-0.000150727, 0.000258861] | 0.601555 | 0.9531 | 0.400 | 0.910 | 0.890 | 否 | rmse_improvement_below_10pct;holm_p_value_not_below_0.05 |

符合全部五项优越性条件的配置数：0。

## 面板层结论

没有配置满足预设优越性规则；完整负面结果与平滑 DGP 结果共同界定适用边界。

## Nuisance 诊断解释

有 4 个配置改善了 nuisance prediction，但未转化为显著的处理效应估计优势。
