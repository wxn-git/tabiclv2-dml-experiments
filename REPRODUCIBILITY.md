# 实验复现指南

本文档说明如何验证代码、重新运行 Stage 1–3B、恢复中断任务并重新生成主要汇总文件。建议先运行 smoke experiment，再决定是否启动完整实验；完整实验包含 TabICLv2 GPU 推理和大量蒙特卡洛重复，耗时较长。

## 1. 已归档运行环境

主要正式实验记录的环境为：

| 项目 | 版本或设备 |
| --- | --- |
| 操作系统 | Windows 11 |
| Python | 3.12.13 |
| NumPy | 2.3.5 |
| pandas | 3.0.1 |
| SciPy | 1.18.0 |
| scikit-learn | 1.9.0 |
| XGBoost | 3.3.0 |
| PyTorch | 2.11.0+cu128 |
| TabICL | 2.1.1 |
| DoubleML | 0.11.3 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB |

机器可读记录见 [`results/published/environment/`](results/published/environment/)。`environment_stage3.json` 中的绝对 Python 路径仅用于记录原实验机器，不应复制到其他电脑。

## 2. 安装

在项目根目录执行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

`.[all]` 会安装传统 learner、TabICLv2、DoubleML、绘图和测试依赖。TabICLv2 第一次运行时可能需要下载预训练权重；权重不属于本仓库。

检查 CUDA：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

没有 NVIDIA GPU 时仍可运行传统 learner 和大部分测试，但无法按原设置复现 TabICLv2 的 GPU 实验。

## 3. 代码正确性验证

运行全部测试：

```powershell
python -m pytest -q
```

归档前的基线结果为：

```text
69 passed, 5 warnings
```

5 条警告来自小规模测试中 MLP 达到 `max_iter=30` 后尚未完全收敛，不是测试失败。

验证本项目的 DML 点估计和标准误是否与 DoubleML 一致：

```powershell
python scripts/validate_doubleml.py
```

已归档验证中，点估计差为 `0.0`，标准误差异约为 `5.55e-17`，见 [`doubleml_validation.json`](results/published/environment/doubleml_validation.json)。

## 4. 计算资源与可比性

正式实验使用以下分工：

- Lasso、随机森林、XGBoost、MLP、ExtraTrees 和传统 ensemble 使用 CPU；
- TabICLv2 使用单个 GPU worker；
- 并行阶段最多使用 8 个 CPU worker 和 1 个 GPU worker；
- 每个 learner 使用相同的模拟数据种子和外层交叉拟合划分。

CPU/GPU 的不同不会改变 Bias、RMSE 或 Coverage 的定义，因此统计准确率可以比较。运行时间同时受到硬件、并行方式、实现和预训练权重缓存影响，只能解释为“当前实验环境下的实际成本”，不能解释为算法在所有机器上的绝对速度。

## 5. 输出和断点续跑

每个模拟任务保存为单独的 JSON。任务身份包含阶段、DGP、样本量、维度、重复编号和 learner。再次运行相同命令时，成功文件会被跳过，因此关机或中断后可以使用原命令继续。

- 正式 JSON：对应阶段的 `results/*_raw/` 或 `results/raw/`；
- nuisance prediction cache：`results/stage3b_cache_*/`；
- worker 日志和状态：`results/logs/`；
- 精简成果：`results/published/`。

只有显式传入 `--retry-failed` 的单进程入口会重试失败任务。恢复前先保留现有输出，不要删除整个结果目录。

## 6. Smoke experiment

先用一个小任务验证传统 learner：

```powershell
python scripts/run_stage1.py --scenarios linear --sample-sizes 500 --dimensions 10 --replications 1 --learners lasso xgboost --fast --output-root results/smoke_raw
```

再单独验证 TabICLv2：

```powershell
python scripts/run_stage1.py --scenarios linear --sample-sizes 500 --dimensions 10 --replications 1 --learners tabiclv2 --fast --output-root results/tabicl_smoke_raw
```

## 7. Stage 1：初步筛选

配置文件：[`configs/stage1.yaml`](configs/stage1.yaml)。固定 `theta0=1`、五折交叉拟合、20 次重复，并覆盖 linear、smooth、tree 和 mixed 四类 DGP。

推荐的并行命令：

```powershell
python scripts/run_stage1_parallel.py --config configs/stage1.yaml --cpu-workers 8 --output-root results/raw --log-dir results/logs/stage1_full
```

生成 Stage 1 总表并选择 Stage 2 配置：

```powershell
python scripts/aggregate_results.py --input results/raw --output results/summary_stage1.csv
python scripts/select_stage2.py --summary results/summary_stage1.csv --output configs/stage2_selected.yaml
```

Stage 1 只用于筛选，不应把 20 次重复的覆盖率当作最终推断结论。

## 8. Stage 2：正式比较

锁定配置：[`configs/stage2_selected.yaml`](configs/stage2_selected.yaml)。Stage 2 包含 7 个数据配置、7 类 learner、100 次重复和五折交叉拟合。

并行运行或断点续跑：

```powershell
python scripts/run_stage2_resume_parallel.py --config configs/stage2_selected.yaml --output-root results/raw --log-dir results/logs/stage2_resume --cpu-workers 8
```

汇总正式结果：

```powershell
python scripts/aggregate_results.py --input results/raw --output results/summary_stage2.csv
```

精简后的 Stage 2 表、中文分析和图表位于 [`results/published/stage2/`](results/published/stage2/)。当前仓库保存了这些分析产物，但尚未包含把 `summary_stage2.csv` 自动转换为所有配对检验表和中文报告的独立脚本；严格复现时应把它视为一个已记录的流程缺口，而不是假装能够一条命令完全再生。

## 9. Stage 3：树状 DGP 初步诊断

配置文件：[`configs/stage3_tree_diagnosis.yaml`](configs/stage3_tree_diagnosis.yaml)。固定 tree、`n=2000`、`p=10`，分别组合 oracle、TabICLv2-1 和 XGBoost 的 `l`/`m`。

```powershell
python scripts/run_stage3_parallel.py --config configs/stage3_tree_diagnosis.yaml --output-root results/stage3_tree_diagnosis_raw --log-dir results/logs/stage3_tree_diagnosis_full --cpu-workers 8 --replications 50
```

Stage 3 的作用是定位问题；正式机制确认由 Stage 3B 完成。

## 10. Stage 3B：论文级机制诊断

配置文件：[`configs/stage3b_tree_publication.yaml`](configs/stage3b_tree_publication.yaml)。三个批次使用不同的 seed namespace，防止筛选和确认共用随机样本。

### Batch A：复现和误差分解

```powershell
python scripts/run_stage3b_batch_a_parallel.py --replications 50 --cpu-workers 8 --cache-root results/stage3b_cache_batch_a --output-root results/stage3b_batch_a_raw --log-dir results/logs/stage3b_batch_a
```

### Batch B：处理模型筛选

```powershell
python scripts/run_stage3b_screen_parallel.py --replications 10 --cpu-workers 8 --output-root results/stage3b_screening_raw --selected-output results/stage3b_screening/selected_models.json --log-dir results/logs/stage3b_screening
```

候选选择只使用可观测的 validation `D` MSE；对真实 `m0` 的 MSE 仅用于模拟诊断，不能用于现实数据中的模型选择。

### Batch C：独立确认

```powershell
python scripts/run_stage3b_parallel.py --replications 50 --cpu-workers 8 --cache-root results/stage3b_cache_confirmation --output-root results/stage3b_confirmation_raw --selected-models results/stage3b_screening/selected_models.json --log-dir results/logs/stage3b_confirmation
```

汇总三个批次：

```powershell
python scripts/aggregate_stage3b.py
```

预期生成：

```text
results/stage3b_analysis/batch_a_summary.csv
results/stage3b_analysis/screening_summary.csv
results/stage3b_analysis/confirmation_summary.csv
results/stage3b_analysis/analysis_report_zh.md
```

### `tree_simple` 轴对齐阈值重跑

配置文件：[`configs/stage3b_tree_simple.yaml`](configs/stage3b_tree_simple.yaml)。该场景保留三个单变量轴对齐阈值，删除原 tree 的乘积交互和非轴对齐边界。旧结果不会被覆盖。

```powershell
python scripts/run_stage3b_batch_a_parallel.py --replications 50 --cpu-workers 8 --stage stage3b_tree_simple_batch_a --seed-namespace stage3b_tree_simple_batch_a --scenario tree_simple --cache-root results/stage3b_tree_simple_cache_batch_a --output-root results/stage3b_tree_simple_batch_a_raw --log-dir results/logs/stage3b_tree_simple_batch_a

python scripts/run_stage3b_screen_parallel.py --config configs/stage3b_tree_simple.yaml --replications 10 --cpu-workers 8 --output-root results/stage3b_tree_simple_screening_raw --selected-output results/stage3b_tree_simple_screening/selected_models.json --log-dir results/logs/stage3b_tree_simple_screening

python scripts/run_stage3b_parallel.py --config configs/stage3b_tree_simple.yaml --replications 50 --cpu-workers 8 --cache-root results/stage3b_tree_simple_cache_confirmation --output-root results/stage3b_tree_simple_confirmation_raw --selected-models results/stage3b_tree_simple_screening/selected_models.json --log-dir results/logs/stage3b_tree_simple_confirmation

python scripts/aggregate_stage3b.py --batch-a-root results/stage3b_tree_simple_batch_a_raw --screening-root results/stage3b_tree_simple_screening_raw --confirmation-root results/stage3b_tree_simple_confirmation_raw --output-root results/stage3b_tree_simple_analysis --title "Stage 3B Tree Simple机制诊断与处理模型筛选结果" --baseline-confirmation-summary results/published/stage3b/confirmation_summary.csv
```

论文级汇总、冻结配置和新旧场景对照位于 [`results/published/stage3b_tree_simple/`](results/published/stage3b_tree_simple/)。

## 11. 恢复 Release 中的原始结果

从 GitHub Release 下载 `tabiclv2-dml-experiments-v0.1-stage3b-raw-results.zip`，在仓库根目录解压。归档保留 `results/...` 相对路径，因此解压后会回到各正式结果目录。

验证下载文件：

```powershell
Get-FileHash tabiclv2-dml-experiments-v0.1-stage3b-raw-results.zip -Algorithm SHA256
```

将结果与 [`ARCHIVE_MANIFEST.md`](ARCHIVE_MANIFEST.md) 中的 SHA-256 比较。二者必须完全相同。

## 12. 已知限制

- 完整实验依赖 TabICLv2 预训练权重和可用 GPU；
- Stage 2 的论文级二次分析尚未完全脚本化；
- Stage 3B 独立确认仅有 50 次重复，不足以生成最终高精度覆盖率表；
- 合成 DGP 结论不能直接外推到真实数据；
- 运行时间结果依赖本次软硬件与并行配置。
