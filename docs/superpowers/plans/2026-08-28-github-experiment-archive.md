# GitHub Experiment Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已完成的 TabICLv2-DML 实验整理为一个安全、可读、可复现且可由用户亲自上传的私有 GitHub 项目。

**Architecture:** 普通 Git 仓库只保存代码、配置、测试、方法文档和精简后的发布结果；完整原始 JSON 与必要诊断缓存进入一个被 Git 忽略的版本化 ZIP，后续由用户上传到 GitHub Release。整理过程不修改原始实验结果，不登录 GitHub，不创建 commit，也不执行 push。

**Tech Stack:** Git 2.55、GitHub CLI 2.98、PowerShell 7、Python 3.12、pytest、Markdown、CSV、ZIP、SHA-256。

## Global Constraints

- 远程仓库名称固定为 `tabiclv2-dml-experiments`，可见性固定为 Private，默认分支为 `main`。
- 首个版本标签固定为 `v0.1-stage3b`。
- 不上传 `.venv`、缓存、日志、模型权重、参考论文 PDF、凭据或烟雾测试结果。
- 不修改 DGP、模型实现、配置含义或实验数字。
- Codex 只整理和验证；用户亲自执行登录、最终暂存检查、commit、push 和 Release 上传。

---

### Task 1: 建立安全的 Git 文件边界

**Files:**
- Create: `.gitignore`
- Create: `results/published/README.md`

**Interfaces:**
- Consumes: 当前项目目录和归档设计文档。
- Produces: 明确的 Git 包含/排除边界，以及发布结果目录说明。

- [ ] **Step 1: 编写 `.gitignore`**

排除虚拟环境、Python 产物、缓存、日志、临时目录、原始结果、中间缓存、模型权重、PDF、压缩包、Office 导师汇报文件和本机秘密文件；显式允许 `results/published/`。

- [ ] **Step 2: 编写发布结果目录说明**

说明 `results/published/` 中的文件均为已有实验输出的只读副本，并给出 Stage 1、Stage 2、Stage 3、Stage 3B、图表和环境记录的目录职责。

- [ ] **Step 3: 验证忽略规则**

Run:

```powershell
git check-ignore -v .venv results/raw 2602.11139v1.pdf
git check-ignore -v results/published/README.md
```

Expected: 前三个路径显示匹配的排除规则；`results/published/README.md` 不被忽略。

### Task 2: 整理可直接阅读的实验成果

**Files:**
- Create: `results/published/stage1/*`
- Create: `results/published/stage2/*`
- Create: `results/published/stage3/README.md`
- Create: `results/published/stage3b/*`
- Create: `results/published/figures/*`
- Create: `results/published/environment/*`

**Interfaces:**
- Consumes: `results/stage1_analysis/`、`results/stage2_analysis/`、`results/stage3b_analysis/`、`results/stage3b_screening/selected_models.json`、根目录汇总文件和环境文件。
- Produces: README 与 RESULTS 可以稳定引用的精简发布结果。

- [ ] **Step 1: 复制 Stage 1 结果**

复制 `results/summary_stage1.csv` 和 `results/stage1_analysis/` 下四个 CSV，不修改源文件。

- [ ] **Step 2: 复制 Stage 2 结果**

复制 `results/summary_stage2.csv` 和 `results/stage2_analysis/` 下的报告、六个 CSV 及图表，不修改源文件。

- [ ] **Step 3: 记录 Stage 3 诊断角色**

创建 `results/published/stage3/README.md`，说明 Stage 3 是树状 DGP 的初步定位实验，完整原始结果位于 Release，正式确认结果见 Stage 3B。

- [ ] **Step 4: 复制 Stage 3B 结果**

复制 `analysis_report_zh.md`、`batch_a_summary.csv`、`screening_summary.csv`、`confirmation_summary.csv` 和 `selected_models.json`。

- [ ] **Step 5: 复制图表与环境信息**

复制 `results/figures/accuracy_cost_pareto.png`、Stage 2 图表、`environment.json`、`environment_stage3.json` 和 `doubleml_validation.json`。

- [ ] **Step 6: 核对副本完整性**

Run:

```powershell
Get-ChildItem results/published -File -Recurse | Measure-Object
Get-FileHash results/stage3b_analysis/confirmation_summary.csv,results/published/stage3b/confirmation_summary.csv -Algorithm SHA256
```

Expected: 发布目录包含全部计划文件；源文件和对应副本的 SHA-256 相同。

### Task 3: 编写项目入口和实验记录

**Files:**
- Modify: `README.md`
- Create: `REPRODUCIBILITY.md`
- Create: `RESULTS.md`

**Interfaces:**
- Consumes: 四阶段配置、现有中文分析报告、汇总 CSV 和设计文档。
- Produces: 新读者理解项目、复现实验和解释结果所需的三个入口文档。

- [ ] **Step 1: 重写项目 README**

包括研究问题、PLR-DML 中 `l(X)` 与 `m(X)` 的通俗定义、比较 learner、四阶段实验路线、主要结论、快速验证、仓库结构和文档导航。

- [ ] **Step 2: 编写完整复现指南**

包括 Python 环境安装、可选 GPU 依赖、CPU/GPU 分工、公平性说明、Stage 1 至 Stage 3B 的准确运行命令、断点续跑机制、汇总命令、原始结果恢复方式和常见故障。

- [ ] **Step 3: 编写结果说明**

按 Stage 1、2、3、3B 记录目的、设计、指标、关键数字、诊断逻辑、结论和结论边界；所有关键数字链接到 `results/published/` 中的 CSV 或报告。

- [ ] **Step 4: 检查文档内部链接**

Run:

```powershell
rg -n "results/|REPRODUCIBILITY|RESULTS|ARCHIVE_MANIFEST" README.md REPRODUCIBILITY.md RESULTS.md
```

Expected: 每个引用路径都存在，README 能导航到复现、结果和归档文档。

### Task 4: 创建原始结果 Release 归档

**Files:**
- Create, ignored: `_release/tabiclv2-dml-experiments-v0.1-stage3b-raw-results.zip`
- Create, ignored: `_release/raw-file-manifest.csv`
- Create: `ARCHIVE_MANIFEST.md`

**Interfaces:**
- Consumes: `results/raw/`、`results/stage3_tree_diagnosis_raw/`、Stage 3B 三个正式 raw 目录和两个正式 nuisance cache 目录。
- Produces: 可上传到 GitHub Release 的完整归档及 Git 内可追溯的校验说明。

- [ ] **Step 1: 生成原始文件清单**

清单记录归档内每个文件的相对路径、字节数和 SHA-256。只纳入正式结果，不纳入 smoke、日志、重启前重复结果或临时缓存。

- [ ] **Step 2: 创建版本化 ZIP**

归档纳入以下目录：

```text
results/raw/
results/stage3_tree_diagnosis_raw/
results/stage3b_batch_a_raw/
results/stage3b_screening_raw/
results/stage3b_confirmation_raw/
results/stage3b_cache_batch_a/
results/stage3b_cache_confirmation/
```

归档同时包含 `raw-file-manifest.csv`。

- [ ] **Step 3: 计算归档 SHA-256**

Run:

```powershell
Get-FileHash _release/tabiclv2-dml-experiments-v0.1-stage3b-raw-results.zip -Algorithm SHA256
```

Expected: 返回一个 64 位 SHA-256 值。

- [ ] **Step 4: 编写 `ARCHIVE_MANIFEST.md`**

记录归档文件名、创建日期、文件数量、解压后总字节数、ZIP 字节数、SHA-256、包含与排除目录、校验命令和未来的 Release 上传命令。

- [ ] **Step 5: 验证 ZIP 内容**

Run:

```powershell
tar -tf _release/tabiclv2-dml-experiments-v0.1-stage3b-raw-results.zip
```

Expected: 只出现七个正式结果目录和 `raw-file-manifest.csv`，没有 `.venv`、日志、PDF、smoke 或凭据。

### Task 5: 完成上传前验证与用户交接

**Files:**
- Modify if needed: `.gitignore`, `README.md`, `REPRODUCIBILITY.md`, `RESULTS.md`, `ARCHIVE_MANIFEST.md`

**Interfaces:**
- Consumes: 整理后的完整项目和 Release 归档。
- Produces: 一份用户可亲自执行的最终上传命令清单。

- [ ] **Step 1: 运行全部自动测试**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Expected: 69 tests passed；允许保留已知的 sklearn MLP 收敛警告，但必须在交接中如实报告。

- [ ] **Step 2: 扫描敏感信息和危险文件**

Run:

```powershell
rg -n -i "api[_-]?key|access[_-]?token|client[_-]?secret|password\s*=|BEGIN .* PRIVATE KEY|github_pat_|ghp_" configs src scripts tests README.md REPRODUCIBILITY.md RESULTS.md
Get-ChildItem -File -Recurse | Where-Object Length -GT 50MB
```

Expected: 待提交范围没有凭据；Git 中没有超过 50 MB 的文件。

- [ ] **Step 3: 检查 Git 候选文件而不暂存**

Run:

```powershell
git status --short --ignored
git ls-files --others --exclude-standard
```

Expected: 候选列表只包含代码、配置、测试、文档和 `results/published/`；`.venv`、原始结果、`_release`、PDF、缓存、日志和导师汇报文件均为 ignored。

- [ ] **Step 4: 向用户交接亲自上传命令**

用户依次执行：

```powershell
gh auth login
$gitHubUser = gh api user --jq .login
$gitHubId = gh api user --jq .id
git config user.name $gitHubUser
git config user.email "$gitHubId+$gitHubUser@users.noreply.github.com"
git add .gitignore README.md REPRODUCIBILITY.md RESULTS.md ARCHIVE_MANIFEST.md pyproject.toml configs src scripts tests docs results/published
git diff --cached --name-only
git commit -m "Archive TabICLv2-DML experiments through Stage 3B"
gh repo create tabiclv2-dml-experiments --private --source . --remote origin --push
git tag -a v0.1-stage3b -m "Experiments completed through Stage 3B"
git push origin v0.1-stage3b
gh release create v0.1-stage3b "_release/tabiclv2-dml-experiments-v0.1-stage3b-raw-results.zip" --title "Stage 3B experiment archive" --notes-file ARCHIVE_MANIFEST.md
```

Expected: 用户本人完成登录、作者身份配置、最终文件检查、commit、push 和 Release 上传。
