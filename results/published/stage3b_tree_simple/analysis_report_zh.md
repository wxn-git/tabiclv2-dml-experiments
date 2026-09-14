# Stage 3B Tree Simple机制诊断与处理模型筛选结果

## Batch A：现有Stage 3A误差分解

| learner_l | learner_m | bias | rmse | coverage | mean_l_mse | mean_m_mse | mean_lm_error_cross | mean_theta_proxy | mean_proxy_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | oracle | -0.0004 | 0.0258 | 0.9200 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.0004 |
| oracle | tabiclv2_1 | -0.0445 | 0.0519 | 0.4600 | 0.0000 | 0.0463 | 0.0000 | 0.9558 | -0.0003 |
| oracle | xgboost | -0.0701 | 0.0746 | 0.1200 | 0.0000 | 0.0746 | 0.0000 | 0.9306 | -0.0007 |
| tabiclv2_1 | oracle | 0.0007 | 0.0266 | 0.9400 | 0.1434 | 0.0000 | 0.0000 | 1.0000 | 0.0007 |
| tabiclv2_1 | tabiclv2_1 | 0.0041 | 0.0283 | 0.9000 | 0.1434 | 0.0463 | 0.0500 | 1.0036 | 0.0005 |
| tabiclv2_1 | xgboost | -0.0294 | 0.0397 | 0.6600 | 0.1434 | 0.0746 | 0.0429 | 0.9705 | 0.0000 |
| xgboost | oracle | 0.0010 | 0.0267 | 0.9200 | 0.1669 | 0.0000 | 0.0000 | 1.0000 | 0.0010 |
| xgboost | tabiclv2_1 | -0.0093 | 0.0286 | 0.8800 | 0.1669 | 0.0463 | 0.0356 | 0.9898 | 0.0009 |
| xgboost | xgboost | -0.0072 | 0.0275 | 0.9200 | 0.1669 | 0.0746 | 0.0667 | 0.9927 | 0.0001 |

## Batch B：处理模型筛选

| candidate | candidate_group | training_target | replications | mean_validation_d_mse | mean_validation_m0_mse | mean_runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| extra_f10_leaf5 | extra_trees | d | 10 | 1.0883 | 0.0942 | 1.0855 |
| extra_f10_leaf2 | extra_trees | d | 10 | 1.1001 | 0.1080 | 1.5061 |
| extra_f10_leaf1 | extra_trees | d | 10 | 1.1112 | 0.1182 | 2.1646 |
| extra_f05_leaf2 | extra_trees | d | 10 | 1.1420 | 0.1451 | 1.0439 |
| extra_f05_leaf1 | extra_trees | d | 10 | 1.1434 | 0.1478 | 1.4880 |
| extra_f05_leaf5 | extra_trees | d | 10 | 1.1642 | 0.1685 | 0.7420 |
| current_xgboost_m0_diagnostic | oracle_target_diagnostic | m0 | 10 | 1.0035 | 0.0051 | 4.6695 |
| tabiclv2_1_m0_diagnostic | oracle_target_diagnostic | m0 | 10 | 1.0037 | 0.0076 | 0.3853 |
| tabiclv2_1 | tab_baseline | d | 10 | 1.0415 | 0.0497 | 0.7695 |
| tabiclv2_8 | tab_baseline | d | 10 | 1.0417 | 0.0511 | 0.7134 |
| current_xgboost | xgb_baseline | d | 10 | 1.0820 | 0.0862 | 4.9977 |
| xgb_d2_lr003_leaf1 | xgboost_tuned | d | 10 | 1.0692 | 0.0758 | 0.2291 |
| xgb_d2_lr005_leaf5 | xgboost_tuned | d | 10 | 1.0847 | 0.0902 | 0.1645 |
| xgb_d3_lr003_leaf5 | xgboost_tuned | d | 10 | 1.1127 | 0.1178 | 0.3342 |
| xgb_d3_lr005_leaf10 | xgboost_tuned | d | 10 | 1.1380 | 0.1399 | 0.2477 |
| xgb_d4_lr003_leaf5 | xgboost_tuned | d | 10 | 1.1475 | 0.1548 | 0.5340 |
| xgb_d5_lr003_leaf10 | xgboost_tuned | d | 10 | 1.1661 | 0.1755 | 0.6619 |

## Batch C：独立确认

| learner_l | learner_m | bias | rmse | coverage | mean_l_mse | mean_m_mse | mean_lm_error_cross | mean_theta_proxy | mean_proxy_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | extra_trees | -0.0862 | 0.0894 | 0.0600 | 0.0000 | 0.0901 | 0.0000 | 0.9174 | -0.0035 |
| oracle | oracle | -0.0048 | 0.0248 | 0.9000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.0048 |
| oracle | tabiclv2_1 | -0.0499 | 0.0557 | 0.3400 | 0.0000 | 0.0469 | 0.0000 | 0.9552 | -0.0051 |
| oracle | xgboost | -0.0739 | 0.0774 | 0.1200 | 0.0000 | 0.0748 | 0.0000 | 0.9304 | -0.0043 |
| oracle | xgboost_tuned | -0.0665 | 0.0705 | 0.1600 | 0.0000 | 0.0659 | 0.0000 | 0.9382 | -0.0046 |
| tabiclv2_1 | extra_trees | -0.0329 | 0.0408 | 0.6800 | 0.1429 | 0.0901 | 0.0588 | 0.9713 | -0.0042 |
| tabiclv2_1 | oracle | -0.0055 | 0.0251 | 0.9400 | 0.1429 | 0.0000 | 0.0000 | 1.0000 | -0.0055 |
| tabiclv2_1 | tabiclv2_1 | -0.0022 | 0.0248 | 0.8800 | 0.1429 | 0.0469 | 0.0508 | 1.0037 | -0.0059 |
| tabiclv2_1 | xgboost | -0.0334 | 0.0408 | 0.7000 | 0.1429 | 0.0748 | 0.0443 | 0.9716 | -0.0050 |
| tabiclv2_1 | xgboost_tuned | -0.0282 | 0.0371 | 0.7400 | 0.1429 | 0.0659 | 0.0416 | 0.9772 | -0.0054 |
| xgboost | extra_trees | -0.0406 | 0.0467 | 0.5200 | 0.1634 | 0.0901 | 0.0494 | 0.9627 | -0.0032 |
| xgboost | oracle | -0.0045 | 0.0243 | 0.9400 | 0.1634 | 0.0000 | 0.0000 | 1.0000 | -0.0045 |
| xgboost | tabiclv2_1 | -0.0154 | 0.0286 | 0.9000 | 0.1634 | 0.0469 | 0.0359 | 0.9895 | -0.0049 |
| xgboost | xgboost | -0.0115 | 0.0256 | 0.9400 | 0.1634 | 0.0748 | 0.0669 | 0.9926 | -0.0041 |
| xgboost | xgboost_tuned | -0.0091 | 0.0250 | 0.9200 | 0.1634 | 0.0659 | 0.0611 | 0.9954 | -0.0045 |

说明：Batch C仍属于50次筛选后确认；论文最终覆盖率表需使用预先锁定配置和新的200至500次重复。

## 与基准场景的独立确认对照

| learner_l | learner_m | bias_baseline | bias_candidate | rmse_baseline | rmse_candidate | coverage_baseline | coverage_candidate | mean_m_mse_baseline | mean_m_mse_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | extra_trees | -0.1706 | -0.0862 | 0.1717 | 0.0894 | 0.0000 | 0.0600 | 0.2014 | 0.0901 |
| oracle | oracle | -0.0036 | -0.0048 | 0.0199 | 0.0248 | 1.0000 | 0.9000 | 0.0000 | 0.0000 |
| oracle | tabiclv2_1 | -0.1565 | -0.0499 | 0.1579 | 0.0557 | 0.0000 | 0.3400 | 0.1817 | 0.0469 |
| oracle | xgboost | -0.1152 | -0.0739 | 0.1170 | 0.0774 | 0.0000 | 0.1200 | 0.1274 | 0.0748 |
| oracle | xgboost_tuned | -0.1191 | -0.0665 | 0.1210 | 0.0705 | 0.0000 | 0.1600 | 0.1321 | 0.0659 |
| tabiclv2_1 | extra_trees | -0.1032 | -0.0329 | 0.1056 | 0.0408 | 0.0200 | 0.6800 | 0.2014 | 0.0901 |
| tabiclv2_1 | oracle | -0.0020 | -0.0055 | 0.0249 | 0.0251 | 0.9200 | 0.9400 | 0.0000 | 0.0000 |
| tabiclv2_1 | tabiclv2_1 | -0.0928 | -0.0022 | 0.0958 | 0.0248 | 0.0600 | 0.8800 | 0.1817 | 0.0469 |
| tabiclv2_1 | xgboost | -0.0612 | -0.0334 | 0.0660 | 0.0408 | 0.2600 | 0.7000 | 0.1274 | 0.0748 |
| tabiclv2_1 | xgboost_tuned | -0.0673 | -0.0282 | 0.0720 | 0.0371 | 0.2000 | 0.7400 | 0.1321 | 0.0659 |
| xgboost | extra_trees | -0.1210 | -0.0406 | 0.1230 | 0.0467 | 0.0000 | 0.5200 | 0.2014 | 0.0901 |
| xgboost | oracle | -0.0030 | -0.0045 | 0.0239 | 0.0243 | 0.9800 | 0.9400 | 0.0000 | 0.0000 |
| xgboost | tabiclv2_1 | -0.1132 | -0.0154 | 0.1154 | 0.0286 | 0.0000 | 0.9000 | 0.1817 | 0.0469 |
| xgboost | xgboost | -0.0494 | -0.0115 | 0.0550 | 0.0256 | 0.4600 | 0.9400 | 0.1274 | 0.0748 |
| xgboost | xgboost_tuned | -0.0573 | -0.0091 | 0.0628 | 0.0250 | 0.3200 | 0.9200 | 0.1321 | 0.0659 |