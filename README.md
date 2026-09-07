# TabICLv2 as a nuisance learner in PLR-DML

本项目研究一个具体问题：在部分线性双重机器学习（PLR-DML）中，用表格基础模型 TabICLv2 替换 Lasso、随机森林、XGBoost、MLP 或集成学习器，能否改善处理效应估计？

答案不是简单的“能”或“不能”。Stage 2 的 100 次蒙特卡洛重复显示，TabICLv2 在平滑非线性设计中降低了处理效应 RMSE，并保持了接近名义水平的覆盖率；在线性设计中与最佳传统方法基本持平；在原始复杂树状阈值设计中明显变差。Stage 3–3B 将该失败定位到处理 nuisance function `m(X)` 的估计误差。Stage 4 又在更规范的轴对齐树状 DGP 上进行冻结配置确认：TabICLv2-1 在 6 个配置中的 5 个取得更低 RMSE，但没有任何配置同时通过预声明的效应量、显著性、覆盖率和完整性门槛，因此不能宣称它在树状 DGP 上优于 tuned-XGBoost。

## DML 在做什么

模拟数据来自部分线性模型：

```text
Y = θ₀D + g₀(X) + ε
D = m₀(X) + V
```

- `Y` 是结果变量；
- `D` 是处理变量；
- `X` 是控制变量；
- `θ₀` 是希望估计的真实处理效应，本实验固定为 `1`；
- `m₀(X) = E[D | X]` 描述哪些样本更可能获得处理；
- `l₀(X) = E[Y | X]` 描述仅根据控制变量能够预测出的平均结果。

DML 先用机器学习估计 `l(X)` 和 `m(X)`，再从 `Y` 和 `D` 中减去这两部分预测，最后使用剩余变化估计 `θ₀`。`l(X)` 和 `m(X)` 不是最终研究对象，因此称为 nuisance functions（干扰函数或辅助函数）。本项目改变的正是这两个预测器。

## 实验路线

| 阶段 | 作用 | 主要设置 |
| --- | --- | --- |
| Stage 1 | 初步筛选 | 4 类 DGP、48 个数据配置、20 次重复 |
| Stage 2 | 正式比较 | 7 个筛选配置、7 类 learner、100 次重复、五折交叉拟合 |
| Stage 3 | 定位树状设计的问题来源 | `l`/`m` 分别使用 oracle、TabICLv2、XGBoost，50 次重复 |
| Stage 3B | 误差分解、候选筛选和独立确认 | 450 + 170 + 750 个正式结果文件 |
| Stage 4 | 轴对齐树状 DGP 的预声明基准 | 24 个筛选配置、6 个冻结确认配置、每个配置 100 次确认重复 |

所有模拟均使用 `θ₀ = 1`。传统模型运行在 CPU，TabICLv2 运行在 GPU；这不影响统计指标的可比性，但运行时间比较只代表当前软硬件配置。

## 主要发现

| 场景 | 结果 |
| --- | --- |
| 平滑非线性 | TabICLv2 相对最佳传统集成模型将 RMSE 降低约 5%–14%，覆盖率为 0.91–0.95 |
| 线性 | TabICLv2、Lasso 和集成模型 RMSE 接近；Lasso 更简单且更快 |
| 原始复杂树状阈值 | TabICLv2 RMSE 明显高于 XGBoost/Lasso；所有方法均出现严重欠覆盖 |
| 树状机制诊断 | 当 `m(X)` 使用 oracle 时，TabICLv2 或 XGBoost 的 `l(X)` 仍能得到接近无偏的估计；当 `m(X)` 由现有 learner 学习时，偏差重新出现 |
| 轴对齐树状基准 | TabICLv2-1 在 6 个冻结配置中的 5 个方向上降低 RMSE，但 `0/6` 同时满足全部预声明优越性条件 |

Stage 3B 的 50 次独立重复主要用于机制确认和模型筛选。Stage 4 随后使用独立随机种子、冻结配置和每个配置 100 次确认重复进行正式比较；它提供了比 Stage 3B 更严格的树状基准，但结果仍不支持 TabICLv2 的确认性优越声明。

详细数字和结论边界见 [`RESULTS.md`](RESULTS.md)。

## 快速验证

需要 Python 3.11 或更高版本：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
python -m pytest -q
```

本次 Stage 4 归档后的完整验证结果为 629 项测试通过，伴随 5 条 sklearn MLP 收敛警告。

运行一个不使用 TabICLv2 的快速 smoke experiment：

```powershell
python scripts/run_stage1.py --scenarios linear --sample-sizes 500 --dimensions 10 --replications 1 --learners lasso xgboost --fast --output-root results/smoke_raw
```

完整复现顺序、GPU 设置、断点续跑和每阶段命令见 [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)。

## 仓库结构

```text
configs/            四阶段实验配置
src/tabdml/         DGP、交叉拟合、DML、learner、诊断和汇总代码
scripts/            实验入口、并行调度、断点续跑和报告脚本
tests/              单元测试与第三方实现一致性测试
docs/               实验设计和执行计划
results/published/  精简后的汇总表、报告、图表和环境记录
```

普通 Git 历史不包含虚拟环境、模型权重、日志、论文 PDF 或上万个原始 JSON。完整原始结果以版本化 ZIP 保存，内容和 SHA-256 见 [`ARCHIVE_MANIFEST.md`](ARCHIVE_MANIFEST.md)。

## 文档导航

- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)：从安装环境到重新运行实验。
- [`RESULTS.md`](RESULTS.md)：Stage 1–4 的结果、解释和论文结论边界。
- [`ARCHIVE_MANIFEST.md`](ARCHIVE_MANIFEST.md)：原始结果压缩包内容与校验方法。
- [`UPLOAD_GUIDE.md`](UPLOAD_GUIDE.md)：由仓库所有者亲自检查、提交和上传的命令。
- [`results/published/`](results/published/)：可直接阅读和引用的精简成果。
- [`docs/superpowers/specs/`](docs/superpowers/specs/)：实验与归档设计记录。

## 当前范围

该仓库记录合成 PLR-DML 实验，不包含真实数据实证分析，也不声称 TabICLv2 是传统 nuisance learner 的通用替代品。当前证据支持“优势依赖 DGP 结构”这一更有限、也更可复核的结论。

## Stage 4：轴对齐树状 DGP 基准

Stage 4 覆盖两个面板、三个树结构，共 24 个预声明筛选配置；分别冻结每个配置的 `l`/`m` XGBoost，共 48 个选择项，再按六个 `panel × structure` 组冻结确认配置（包括筛选阶段 TabICLv2 落败的组）。确认阶段每个冻结配置运行 100 次独立重复，共生成 6,000 条 DML 结果；发布门禁验证全部记录完整、无失败、无 OOM、无静默 fallback。

三种运行模式必须分开：

- 实现 smoke：各阶段 `--replications 1 --fast`，只检查流程；选择器和分析也必须带 `--fast`、正确的 `--tuned-models`。
- 独立预检：正式 10 次 tuning 和 20 次 screening 冻结后，确认阶段使用 `--preflight`，运行完整模型的 5 次独立抽样；`--replications 5` 本身不是预检。
- 正式实验：10 次 tuning、20 次 screening、100 次 confirmation，不使用 `--fast` 或 `--preflight`；昂贵运行需单独批准。

Windows 使用短根目录，例如 `results/s4s0905/{tr,sr,cr,cache,an}`，避免长任务键触发 MAX_PATH。完整、可断点续跑的 PowerShell 命令及正式发布门禁见 [复现说明第 13 节](REPRODUCIBILITY.md#13-stage-4轴对齐树状基准)。最多一个 GPU worker 和八个 CPU workers。

六项主要比较的预声明规则要求同时满足：RMSE 至少改善 10%、六项比较的 Holm 校正 `p < 0.05`、TabICLv2 coverage 不比 tuned-XGBoost 低超过 0.05、TabICLv2 coverage 至少为 0.90，以及结果完整。TabICLv2-1 在 5/6 个配置中取得方向性 RMSE 改善，改善幅度为 5.07%–19.25%；其中 3/6 达到 10% 的效应量门槛，但 `0/6` 通过全部规则。最接近通过的是标准面板的 `tree_hierarchical`：RMSE 改善 19.25%，未校正配对 `p=0.0083`，但 Holm `p=0.0501` 且 coverage 为 0.89。

这意味着 Stage 4 的正确结论是“存在值得继续研究的方向性信号，但确认性证据不足”，而不是“TabICLv2 已在树状 DGP 上胜过 XGBoost”。完整数字、nuisance 诊断与结论边界见 [`RESULTS.md`](RESULTS.md#9-stage-4轴对齐树状-dgp-确认性基准)，冻结发布产物见 [`results/published/stage4_tree_benchmark/`](results/published/stage4_tree_benchmark/)。
