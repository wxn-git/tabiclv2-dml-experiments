# Stage 4 Results Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已通过发布门禁的 Stage 4 正式结果准确写入项目总览与结果报告，并验证文档改动没有破坏项目。

**Architecture:** 以 `results/published/stage4_tree_benchmark/` 中的发布产物为唯一数字来源。`README.md` 只保留面向读者的研究路线和核心结论，`RESULTS.md` 记录六项冻结比较、预声明门槛与论文表述边界。

**Tech Stack:** Markdown、CSV/JSON 发布产物、Git、pytest。

## Global Constraints

- 不改写 Stage 1–3B 的历史结果。
- 明确区分“方向性改善”和“满足预声明优越性规则”。
- 保留 `0/6` 配置通过全部门槛这一负面确认性结论。
- 不提交 `artifacts/` 中的临时 Word 生成文件。
- 本任务不执行 Git commit 或 push。

---

### Task 1: 更新正式结果文档

**Files:**
- Modify: `README.md`
- Modify: `RESULTS.md`

**Interfaces:**
- Consumes: `results/published/stage4_tree_benchmark/analysis_report_zh.md`、`primary_paired_comparisons.csv` 与 `manifest.json`
- Produces: 面向仓库读者的 Stage 1–4 连贯实验叙述

- [x] **Step 1: 更新 README 的实验路线、核心发现和 Stage 4 状态**
- [x] **Step 2: 在 RESULTS 中加入 Stage 4 的设计、六项比较、诊断与结论边界**
- [x] **Step 3: 检查 Markdown 链接和 Git 差异**

### Task 2: 验证项目

**Files:**
- Test: `tests/test_stage4_analysis.py`
- Test: `tests/test_stage4_publish.py`
- Test: `tests/`

**Interfaces:**
- Consumes: 更新后的文档和当前代码库
- Produces: Stage 4 测试与完整测试结果

- [x] **Step 1: 运行 Stage 4 分析和发布测试**
- [x] **Step 2: 运行完整 pytest 测试集**
- [x] **Step 3: 运行 `git diff --check` 并汇报待提交文件**
