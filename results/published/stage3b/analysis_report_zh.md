# Stage 3B Tree机制诊断与处理模型筛选结果

## Batch A：现有Stage 3A误差分解

| learner_l | learner_m | bias | rmse | coverage | mean_l_mse | mean_m_mse | mean_lm_error_cross | mean_theta_proxy | mean_proxy_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | oracle | -0.0018 | 0.0260 | 0.9200 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.0018 |
| oracle | tabiclv2_1 | -0.1546 | 0.1562 | 0.0000 | 0.0000 | 0.1810 | 0.0000 | 0.8468 | -0.0014 |
| oracle | xgboost | -0.1147 | 0.1168 | 0.0000 | 0.0000 | 0.1264 | 0.0000 | 0.8878 | -0.0026 |
| tabiclv2_1 | oracle | 0.0001 | 0.0263 | 0.8400 | 0.2351 | 0.0000 | 0.0000 | 1.0000 | 0.0001 |
| tabiclv2_1 | tabiclv2_1 | -0.0925 | 0.0957 | 0.0200 | 0.2351 | 0.1810 | 0.0716 | 0.9074 | 0.0000 |
| tabiclv2_1 | xgboost | -0.0616 | 0.0653 | 0.2000 | 0.2351 | 0.1264 | 0.0582 | 0.9395 | -0.0011 |
| xgboost | oracle | 0.0004 | 0.0283 | 0.9000 | 0.2771 | 0.0000 | 0.0000 | 1.0000 | 0.0004 |
| xgboost | tabiclv2_1 | -0.1116 | 0.1148 | 0.0200 | 0.2771 | 0.1810 | 0.0486 | 0.8880 | 0.0004 |
| xgboost | xgboost | -0.0487 | 0.0543 | 0.5400 | 0.2771 | 0.1264 | 0.0725 | 0.9522 | -0.0008 |

## Batch B：处理模型筛选

| candidate | candidate_group | training_target | replications | mean_validation_d_mse | mean_validation_m0_mse | mean_runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| extra_f10_leaf5 | extra_trees | d | 10 | 1.1872 | 0.2078 | 1.0883 |
| extra_f10_leaf2 | extra_trees | d | 10 | 1.1981 | 0.2184 | 1.5857 |
| extra_f10_leaf1 | extra_trees | d | 10 | 1.2109 | 0.2290 | 2.3874 |
| extra_f05_leaf1 | extra_trees | d | 10 | 1.2159 | 0.2403 | 1.3545 |
| extra_f05_leaf2 | extra_trees | d | 10 | 1.2199 | 0.2408 | 1.0602 |
| extra_f05_leaf5 | extra_trees | d | 10 | 1.2446 | 0.2666 | 0.8093 |
| current_xgboost_m0_diagnostic | oracle_target_diagnostic | m0 | 10 | 1.0168 | 0.0194 | 3.6566 |
| tabiclv2_1_m0_diagnostic | oracle_target_diagnostic | m0 | 10 | 1.0806 | 0.0919 | 0.3720 |
| tabiclv2_8 | tab_baseline | d | 10 | 1.1721 | 0.1949 | 0.6903 |
| tabiclv2_1 | tab_baseline | d | 10 | 1.1724 | 0.1932 | 0.7147 |
| current_xgboost | xgb_baseline | d | 10 | 1.1298 | 0.1405 | 3.7403 |
| xgb_d2_lr003_leaf1 | xgboost_tuned | d | 10 | 1.1335 | 0.1468 | 0.1951 |
| xgb_d2_lr005_leaf5 | xgboost_tuned | d | 10 | 1.1358 | 0.1451 | 0.1501 |
| xgb_d3_lr003_leaf5 | xgboost_tuned | d | 10 | 1.1455 | 0.1535 | 0.2845 |
| xgb_d3_lr005_leaf10 | xgboost_tuned | d | 10 | 1.1584 | 0.1682 | 0.2114 |
| xgb_d4_lr003_leaf5 | xgboost_tuned | d | 10 | 1.1798 | 0.1841 | 0.4257 |
| xgb_d5_lr003_leaf10 | xgboost_tuned | d | 10 | 1.1995 | 0.2066 | 0.5492 |

## Batch C：独立确认

| learner_l | learner_m | bias | rmse | coverage | mean_l_mse | mean_m_mse | mean_lm_error_cross | mean_theta_proxy | mean_proxy_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | extra_trees | -0.1706 | 0.1717 | 0.0000 | 0.0000 | 0.2014 | 0.0000 | 0.8324 | -0.0030 |
| oracle | oracle | -0.0036 | 0.0199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.0036 |
| oracle | tabiclv2_1 | -0.1565 | 0.1579 | 0.0000 | 0.0000 | 0.1817 | 0.0000 | 0.8464 | -0.0029 |
| oracle | xgboost | -0.1152 | 0.1170 | 0.0000 | 0.0000 | 0.1274 | 0.0000 | 0.8871 | -0.0023 |
| oracle | xgboost_tuned | -0.1191 | 0.1210 | 0.0000 | 0.0000 | 0.1321 | 0.0000 | 0.8834 | -0.0025 |
| tabiclv2_1 | extra_trees | -0.1032 | 0.1056 | 0.0200 | 0.2428 | 0.2014 | 0.0795 | 0.8985 | -0.0018 |
| tabiclv2_1 | oracle | -0.0020 | 0.0249 | 0.9200 | 0.2428 | 0.0000 | 0.0000 | 1.0000 | -0.0020 |
| tabiclv2_1 | tabiclv2_1 | -0.0928 | 0.0958 | 0.0600 | 0.2428 | 0.1817 | 0.0739 | 0.9089 | -0.0017 |
| tabiclv2_1 | xgboost | -0.0612 | 0.0660 | 0.2600 | 0.2428 | 0.1274 | 0.0593 | 0.9396 | -0.0009 |
| tabiclv2_1 | xgboost_tuned | -0.0673 | 0.0720 | 0.2000 | 0.2428 | 0.1321 | 0.0571 | 0.9338 | -0.0011 |
| xgboost | extra_trees | -0.1210 | 0.1230 | 0.0000 | 0.2830 | 0.2014 | 0.0590 | 0.8815 | -0.0025 |
| xgboost | oracle | -0.0030 | 0.0239 | 0.9800 | 0.2830 | 0.0000 | 0.0000 | 1.0000 | -0.0030 |
| xgboost | tabiclv2_1 | -0.1132 | 0.1154 | 0.0000 | 0.2830 | 0.1817 | 0.0507 | 0.8892 | -0.0024 |
| xgboost | xgboost | -0.0494 | 0.0550 | 0.4600 | 0.2830 | 0.1274 | 0.0737 | 0.9524 | -0.0018 |
| xgboost | xgboost_tuned | -0.0573 | 0.0628 | 0.3200 | 0.2830 | 0.1321 | 0.0695 | 0.9447 | -0.0020 |

说明：Batch C仍属于50次筛选后确认；论文最终覆盖率表需使用预先锁定配置和新的200至500次重复。