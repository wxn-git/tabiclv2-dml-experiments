# Raw-result archive manifest

本文件描述 `v0.1-stage3b` GitHub Release 使用的完整原始结果归档。ZIP 保存在本机 `_release/` 中，并被 `.gitignore` 排除；它不进入普通 Git 历史。

## Archive identity

| 字段 | 值 |
| --- | --- |
| 文件名 | `tabiclv2-dml-experiments-v0.1-stage3b-raw-results.zip` |
| 创建日期 | 2026-08-28 |
| ZIP 文件数 | 13,181 |
| 实验/缓存文件数 | 13,180 |
| ZIP 字节数 | 14,976,346 |
| 解压后总字节数 | 18,812,217 |
| SHA-256 | `e5b94d51b71110a437433acf72be8fc720358a86a86b0667ddc47df6080602b7` |

解压后总字节数包含 `raw-file-manifest.csv`。该 CSV 为 13,180 个实验或缓存文件逐一记录相对路径、字节数和 SHA-256。

## Included data

| 归档内目录 | 文件数 | 字节数 | 作用 |
| --- | ---: | ---: | --- |
| `results/raw/` | 10,660 | 8,501,000 | Stage 1 和 Stage 2 正式 JSON |
| `results/stage3_tree_diagnosis_raw/` | 450 | 404,042 | Stage 3 的 9 个 learner 组合 × 50 次重复 |
| `results/stage3b_batch_a_raw/` | 450 | 643,271 | Stage 3B Batch A 误差分解 |
| `results/stage3b_screening_raw/` | 170 | 149,651 | Stage 3B Batch B 候选筛选 |
| `results/stage3b_confirmation_raw/` | 750 | 1,032,743 | Stage 3B Batch C 独立确认 |
| `results/stage3b_cache_batch_a/` | 300 | 2,377,470 | Batch A out-of-fold nuisance predictions |
| `results/stage3b_cache_confirmation/` | 400 | 3,719,889 | Batch C out-of-fold nuisance predictions |
| `raw-file-manifest.csv` | 1 | 1,984,151 | 文件级字节数和 SHA-256 清单 |

## Excluded data

归档明确排除：

- `.venv` 和所有第三方依赖；
- TabICLv2 预训练权重；
- worker 日志和状态文件；
- smoke、startup check 和临时结果；
- `raw_before_restart_*` 中的重启前重复文件；
- Stage 3B smoke cache；
- 两篇参考论文 PDF 和导师汇报 Word 文件；
- 已在普通 Git 中保存的精简汇总和图表。

## Validation performed

创建归档前后完成了以下检查：

- 12,480 个正式 JSON 均可解析且 `status == "success"`；
- 700 个 `.npz` nuisance cache 均可由 NumPy 打开；
- ZIP 中没有 `.venv`、日志、smoke、PDF、Word 或重启前重复结果；
- ZIP 中 13,180 个实验/缓存文件逐一与 `raw-file-manifest.csv` 比较长度和 SHA-256；
- 文件级校验结果为 13,180 个匹配、0 个不匹配。

## Verify the local or downloaded ZIP

在 PowerShell 中执行：

```powershell
Get-FileHash "_release\tabiclv2-dml-experiments-v0.1-stage3b-raw-results.zip" -Algorithm SHA256
```

结果必须为：

```text
E5B94D51B71110A437433ACF72BE8FC720358A86A86B0667DDC47DF6080602B7
```

查看归档内容：

```powershell
tar -tf "_release\tabiclv2-dml-experiments-v0.1-stage3b-raw-results.zip"
```

## Upload after the Git repository is pushed

由仓库所有者本人执行：

```powershell
gh release create v0.1-stage3b "_release\tabiclv2-dml-experiments-v0.1-stage3b-raw-results.zip" --title "Stage 3B experiment archive" --notes-file ARCHIVE_MANIFEST.md
```

上传后可打开 Release 检查：

```powershell
gh release view v0.1-stage3b --web
```

下载后的 ZIP 必须再次计算 SHA-256，并与本文件记录完全一致。
