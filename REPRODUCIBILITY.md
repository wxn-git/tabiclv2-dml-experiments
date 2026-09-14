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

## 13. Stage 4：轴对齐树状基准

配置为 `configs/stage4_tree_benchmark.yaml`：固定 `theta0=1`、五折、24 个配置；每个配置有六种同方法估计和四种交叉/oracle 诊断组合，共十种 DML 配对。以下命令均在仓库根目录执行，每条命令退出码为 0 后才继续下一条。不要覆盖历史 Stage 1–3B 或已有他人运行目录。这里的命令是复现流程，不是已完成正式实验的声明。

### 13.1 一次快速实现 smoke（不是五次预检）

使用独立短根目录 `results/s4s0905`；若该目录正被使用，先改成另一个短名称，不要同时启动重复流程。Windows 的长任务键可能达到 MAX_PATH，因此不要把原始输出放入长目录名。以下各个模型阶段均显式使用 `--replications 1 --fast`。`--fast` 使用缩减模型，不能解释为完整模型性能。

```powershell
$py = ".\.venv\Scripts\python.exe"
$cfg = "configs/stage4_tree_benchmark.yaml"
$s = "results/s4s0905"

# 仅环境准备检查；完整 TabICLv2 运行还要求本地权重可用。
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
nvidia-smi
& $py scripts/check_stage4_tree_structures.py --n 200000 --seed 20260903 --output-dir "$s/st"
& $py scripts/run_stage4_parallel.py --config $cfg --phase tuning --replications 1 --fast --cpu-workers 8 --output-root "$s/tr" --log-dir "$s/log/tuning"
& $py scripts/run_stage4_tuning.py --config $cfg --output-root "$s/tr" --selected-output "$s/tuned.json" --replications 1 --select --fast
& $py scripts/run_stage4_parallel.py --config $cfg --phase screening --replications 1 --fast --cpu-workers 8 --tuned-models "$s/tuned.json" --cache-root "$s/cache" --output-root "$s/sr" --log-dir "$s/log/screening"
& $py scripts/select_stage4_confirmation.py --config $cfg --screening-root "$s/sr" --tuned-models "$s/tuned.json" --expected-replications 1 --fast --output "$s/cells.json"
& $py scripts/run_stage4_parallel.py --config $cfg --phase confirmation --replications 1 --fast --cpu-workers 8 --tuned-models "$s/tuned.json" --selected-cells "$s/cells.json" --cache-root "$s/cache" --output-root "$s/cr" --log-dir "$s/log/confirmation"
& $py scripts/analyze_stage4.py --config $cfg --screening-root "$s/sr" --confirmation-root "$s/cr" --tuned-models "$s/tuned.json" --selected-cells "$s/cells.json" --fast --output-dir "$s/an"
& $py scripts/environment_report.py --output "$s/an/environment.json"
```

`run_stage4_tuning.py --select` 也遍历任务：完整成功记录会被复用；若上一步尚未结束或有缺失，它可能执行拟合。因此只能在 tuning 并行流程成功结束后运行，不要把它当作无条件只读命令。

预期检查项（必须以实际日志验证，而不是仅看文件数）：结构诊断 12 行；tuning 288 条成功记录与 48 个冻结项；screening 288 个 cache 任务、240 条 DML、24 配置；六个冻结确认配置；confirmation 72 个 cache 任务、60 条 DML；每个 phase 的 `progress.json` 无失败或未完成项，所有 workers/compose 退出码 0。检查 stderr 中未解释的 traceback、OOM 和 fallback。一次 smoke 的主要比较标记 `inference_status=implementation_smoke`，配对 p 值、Holm p 值和差值区间为缺失，不作统计推断，不发布为最终证据。

### 13.2 正式 tuning/screening（10/20 次，批准后运行）

另用 `results/s4f0905`。不要把 smoke 的 frozen tuning 或 selected cells 混入这里；把 `--fast` 去掉并重命名目录不能把 smoke 转成正式结果。

```powershell
$py = ".\.venv\Scripts\python.exe"
$cfg = "configs/stage4_tree_benchmark.yaml"
$f = "results/s4f0905"
& $py scripts/check_stage4_tree_structures.py --n 200000 --seed 20260903 --output-dir "$f/st"
& $py scripts/run_stage4_parallel.py --config $cfg --phase tuning --replications 10 --cpu-workers 8 --output-root "$f/tr" --log-dir "$f/log/tuning"
& $py scripts/run_stage4_tuning.py --config $cfg --output-root "$f/tr" --selected-output "$f/tuned.json" --replications 10 --select
& $py scripts/run_stage4_parallel.py --config $cfg --phase screening --replications 20 --cpu-workers 8 --tuned-models "$f/tuned.json" --cache-root "$f/cache" --output-root "$f/sr" --log-dir "$f/log/screening"
& $py scripts/select_stage4_confirmation.py --config $cfg --screening-root "$f/sr" --tuned-models "$f/tuned.json" --expected-replications 20 --profile full --output "$f/cells.json"
```

tuning 必须覆盖 `24 × 2 targets × 6 candidates × 10 = 2880` 条记录，依据观测 Y/D validation MSE 选取 winners；truth MSE 只用于诊断。screening 必须覆盖 `24 × 10 method pairs × 20 = 4800` 条 DML。每个 `panel × structure` 组都选一个配置，包括负面结果，不能事后只保留有利配置。

### 13.3 五次独立完整模型预检

在上述正式 tuning 和 screening 冻结后运行（不是使用 fast 冻结项）。独立根目录 `results/s4p0905` 避免与正式输出混合：

```powershell
$p = "results/s4p0905"
& $py scripts/run_stage4_parallel.py --config $cfg --phase confirmation --preflight --cpu-workers 8 --tuned-models "$f/tuned.json" --selected-cells "$f/cells.json" --cache-root "$p/cache" --output-root "$p/cr" --log-dir "$p/log/confirmation"
```

`--preflight` 强制五次完整模型抽样，使用 `stage4_tree_confirmation_preflight` stage 和独立 seed namespace；禁止 `--fast`。预期 `6 × 10 × 5 = 300` 条 DML。单独 `--replications 5` 只会运行正式 namespace 的前五次，不是独立预检。当前正式 analysis 和 publisher 拒绝预检记录；预检只检查完成、资源和错误状态，不生成正式结论。

### 13.4 一百次正式 confirmation 与分析

预检通过且正式运行获批后，回到 `$f`，不要使用 `$p/cr`：

```powershell
& $py scripts/run_stage4_parallel.py --config $cfg --phase confirmation --replications 100 --cpu-workers 8 --tuned-models "$f/tuned.json" --selected-cells "$f/cells.json" --cache-root "$f/cache" --output-root "$f/cr" --log-dir "$f/log/confirmation"
& $py scripts/analyze_stage4.py --config $cfg --screening-root "$f/sr" --confirmation-root "$f/cr" --tuned-models "$f/tuned.json" --selected-cells "$f/cells.json" --profile full --output-dir "$f/an"
& $py scripts/environment_report.py --output "$f/an/environment.json"
```

必须有 `6 × 10 × 100 = 6000` 条完整配对成功记录，其中每个确认配置的六种同方法估计都各有 100 次。主要检验是六项 TabICLv2-1 对 tuned-XGBoost 的配对平方误差比较，固定 `alpha=0.05`、Holm 校正和 exact coverage intervals；不允许看结果后更改规则。

### 13.5 文件布局与发布门禁

`an` 中的分析输出始终放在一起；与早期计划中拆分 screening/confirmation CSV 的路径示例不同，当前发布器默认也从一个 `stage4_tree_confirmation` 目录读取它们，不要求人为重复复制：

```text
an/
  screening_summary.csv
  screening_cell_ranking.csv       # 完整 24 配置排名，不能省略
  confirmation_summary.csv
  primary_paired_comparisons.csv
  coverage_diagnostics.csv
  nuisance_diagnostics.csv
  analysis_report_zh.md
  environment.json                # 单独 environment_report 命令生成
  figures/dml_rmse_by_panel.png
  figures/nuisance_mse_by_panel.png
  figures/coverage_by_panel.png
```

以下是仅供完整正式结果使用的门禁，不要在 smoke/preflight 上运行：

```powershell
& $py scripts/publish_stage4.py --results-root $f --config $cfg --structure-dir "$f/st" --tuned-models "$f/tuned.json" --selected-cells "$f/cells.json" --tuning-root "$f/tr" --screening-root "$f/sr" --confirmation-root "$f/cr" --analysis-dir "$f/an" --expected-replications 100 --destination results/published/stage4_tree_benchmark
```

使用计划中的默认长目录布局时，可用下列等价入口（Windows 原始任务仍推荐短路径覆盖）：

```powershell
& $py scripts/publish_stage4.py --results-root results --expected-replications 100 --destination results/published/stage4_tree_benchmark
```

默认路径为 `stage4_tree_structure_checks/structure_checks.{json,csv}`、`stage4_tree_tuning/selected_xgboost.json`、`stage4_tree_screening/selected_confirmation_cells.json`、三个 `stage4_tree_{tuning,screening,confirmation}_raw/`，以及同一个 `stage4_tree_confirmation/` 分析目录，均相对 `--results-root`。`--config` 默认使用仓库配置；显式路径覆盖相对当前工作目录，命令须从仓库根目录执行。

发布器先校验所有必需文件、固定配置、完整 full-profile 原始任务身份/seed/model hashes、重复/缺失/fallback，再从 tuning raw 重算 48 个 winners，从 screening 重算选择，从 screening/confirmation 重算全部六 CSV、报告和三图。原始 NaN/Infinity、非正式推断、陈旧汇总或图均拒绝。CPU 的缺失 GPU telemetry 属于允许的缺失值，不等于无效推断。文件逐字节核对，因此跨绘图库/字体版本的旧图需要用当前环境重跑 analysis，之后重新生成 environment report。

验证成功后才建立临时同级目录，复制 16 个精简文件（含 config、structure、冻结项、环境、分析），写入 SHA-256 manifest；不复制 raw/cache/log。manifest 对原始输入文件名与 SHA-256 的有序集合另存摘要和计数，不把所有 raw 文件展开进精简发布。输入在验证/复制期间改变会拒绝安装。它验证内部一致性，不证明原始记录绝无伪造，也不代替独立保管原始实验数据。

目标不存在时通过一次目录 rename 安装。已存在目标默认拒绝；只有显式 `--replace` 且目标为未修改的同类 Stage 4 publication 才可替换。已有目录替换是 backup rename → install rename（不是 Windows 上单次无缝目录交换）；安装异常会回滚。崩溃中断时保留的 `.backup-*` 和 `.publish.lock` 必须人工检查，确认没有运行中的发布器后恢复，不要盲删。不要用 `--replace` 更新历史 Stage 1–3B。

### 13.6 断点续跑、日志和 Git 边界

重复执行完全相同的命令会复用成功的原子 JSON/cache；不要更改配置、profile、冻结项、seed namespace 或输出根后期望复用旧任务。fast/full 具有不同任务身份，confirmation preflight 使用额外 namespace；正式 tuning/screening/confirmation 分别使用配置声明的三个 namespace。失败/OOM 记录不会静默成为成功；确认原因后对 `run_stage4_parallel.py`（以及支持此参数的底层 runner/compose）显式添加 `--retry-failed`。单纯重跑选择器或 analysis 不会修复缺失的实验任务。analysis 会整体替换输出目录，因此每次重跑后再次写入 `environment.json`。

并行流程日志与 `progress.json` 位于各个显式 `--log-dir`；不要仅依据 CLI 的一个退出码跳过逐 phase 的计数和 stderr 检查。运行 `--dry-run` 可检查已具备所需冻结输入的调度命令而不启动 workers。

环境报告保留原有 `collect_environment()` 行为：`cuda` 字段实际记录 torch 包版本，不能当作单独的 CUDA runtime 版本或 GPU 可用性证明；GPU/driver 信息来自 `nvidia-smi`，CUDA 可用性以运行前检查为准。

`.gitignore` 只开放 Stage 4 精简目录中的明确文件名，包含六 CSV、三图；raw/cache/log、`.superpowers/` 和 `results/s4s*`/`s4p*`/`s4f*` 工作目录仍被忽略。Git 可跟踪不等于允许作为正式证据；提交生成结果须另行检查和决定。不要使用 `git add -f` 将 smoke 原始文件纳入发布。

```powershell
& $py -m pytest tests/test_stage4_publish.py tests/test_environment_report.py -q
& $py -m pytest -q
git check-ignore results/stage4_tree_tuning_raw/example.json results/stage4_tree_cache/example.npz results/s4s0905/sr/example.json
git check-ignore results/stage4_tree_confirmation/screening_cell_ranking.csv
git diff --check
git status --short
```

最后一项 `check-ignore` 应无输出（退出码 1），表明精简排名文件可跟踪；前一项应输出所有被忽略路径。真实 smoke 的最终完成证据与耗时由实际运行日志另行记录，不能从合成单元测试推断成功，也不能据此替代正式 10/20/100 次实验。
