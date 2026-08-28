# GitHub 首次上传：逐步操作与解读

这是一份面向第一次使用 Git 和 GitHub CLI 的操作教程。请在 PowerShell 中按顺序执行，不要一次复制整份文档。每完成一步，先看懂输出，再进入下一步。

本教程的目标是：

1. 登录你自己的 GitHub 账号；
2. 使用隐私邮箱记录提交作者；
3. 检查真正准备上传的文件；
4. 创建本地第一次提交；
5. 创建私有 GitHub 仓库并推送；
6. 把完整原始结果作为 GitHub Release 附件上传；
7. 验证远程仓库和压缩包都正确。

Codex 已经完成文件整理和安全检查，但没有替你执行 `git add`、`git commit`、GitHub 登录或推送。

## 一、先理解五个基本概念

### 1. Git

Git 是电脑上的版本管理工具。即使不联网，它也能记录项目在不同时间的状态。

### 2. GitHub

GitHub 是存放 Git 仓库的在线平台。Git 和 GitHub 不是同一个东西：Git 负责本地版本，GitHub 负责远程保存和协作。

### 3. 暂存区

暂存区可以理解为“下一次提交准备装进箱子的文件”。

```text
普通文件 --git add--> 暂存区 --git commit--> 本地版本历史
```

文件进入暂存区不等于已经上传，也不等于已经提交。

### 4. Commit

Commit 是本地版本快照。它会记录：

- 哪些文件发生了变化；
- 作者名和作者邮箱；
- 提交时间；
- 一句修改说明；
- 唯一的提交编号。

### 5. Push

Push 是把本地 commit 发送到 GitHub。只有执行 push 后，内容才真正出现在远程仓库中。

完整流程是：

```text
项目文件
   ↓ git add
暂存区
   ↓ git commit
本地版本历史
   ↓ git push
GitHub 私有仓库
```

## 二、当前项目处于什么状态

当前项目已经：

- 初始化为 Git 仓库；
- 使用 `main` 分支；
- 建立 `.gitignore`；
- 整理好代码、配置、测试、文档和精简结果；
- 生成 GitHub Release 原始结果 ZIP；
- 通过测试、敏感信息和大文件检查。

当前项目还没有：

- 登录 GitHub；
- 配置 Git 作者；
- 将文件放入暂存区；
- 创建第一次 commit；
- 创建远程仓库；
- 执行 push。

这意味着现在仍然是最安全的学习阶段。即使暂存时出错，也可以在提交前撤回。

## 三、打开 PowerShell 并进入项目

建议重新打开一个 PowerShell 窗口，让新安装的 GitHub CLI PATH 生效。

执行：

```powershell
cd "C:\Users\Stern\Desktop\论文实验"
```

### 这条命令是什么意思

- `cd` 是 change directory，表示切换文件夹；
- 引号用于保护包含中文或空格的路径；
- 后续 Git 命令必须在项目目录中执行。

确认当前目录：

```powershell
Get-Location
```

正常情况下应显示：

```text
C:\Users\Stern\Desktop\论文实验
```

检查三个工具：

```powershell
git --version
gh --version
python --version
```

其中：

- `git` 用于本地版本管理；
- `gh` 是 GitHub CLI，用于登录和操作 GitHub；
- `python` 用于运行实验。

如果出现“不是 Git 仓库”，通常是因为 PowerShell 没有进入正确目录，重新执行 `cd` 即可。

## 四、登录 GitHub

执行：

```powershell
gh auth login
```

终端会逐步提问，推荐选择：

```text
Where do you use GitHub?                 GitHub.com
What is your preferred protocol?         HTTPS
Authenticate Git with GitHub credentials Yes
How would you like to authenticate?      Login with a web browser
```

### 浏览器登录过程

GitHub CLI 会显示一串临时设备代码，并要求按 Enter 打开浏览器。浏览器中：

1. 登录你自己的 GitHub 账号；
2. 输入终端显示的设备代码；
3. 检查申请授权的是 GitHub CLI；
4. 点击授权；
5. 回到 PowerShell。

不要把密码、验证码或 Token 写入项目，也不要发送给其他人。

检查登录状态：

```powershell
gh auth status
```

正常输出应包含：

- `Logged in to github.com`；
- 你的 GitHub 用户名；
- Git 协议为 `https`；
- Token scopes 信息。

这一步只证明 GitHub CLI 已登录，还没有上传任何项目文件。

## 五、配置提交作者和隐私邮箱

### 为什么还要配置作者

GitHub 登录回答“谁有权限上传”，Git 作者配置回答“这次 commit 显示是谁创建的”。二者是不同设置。

如果 commit 使用真实私人邮箱，而 GitHub 开启了：

```text
Block command line pushes that expose my email
```

GitHub 可能拒绝 push。正确做法是使用 GitHub 的 `noreply` 隐私邮箱。

### 1. 读取登录账号信息

```powershell
$gitHubUser = gh api user --jq .login
$gitHubId = gh api user --jq .id
$noreplyEmail = "$gitHubId+$gitHubUser@users.noreply.github.com"
```

查看三个变量：

```powershell
$gitHubUser
$gitHubId
$noreplyEmail
```

隐私邮箱通常类似：

```text
12345678+username@users.noreply.github.com
```

建议同时打开 GitHub：

```text
Settings → Emails
```

确认页面显示的隐私邮箱与 `$noreplyEmail` 一致。如果页面给出的地址不同，以 GitHub 页面显示的准确地址为准：

```powershell
$noreplyEmail = "在这里粘贴 GitHub 页面显示的 noreply 邮箱"
```

### 2. 只为当前项目配置作者

```powershell
git config --local user.name "$gitHubUser"
git config --local user.email "$noreplyEmail"
```

这里使用 `--local`，表示只影响当前论文实验仓库，不修改电脑上其他 Git 项目的作者信息。

检查配置：

```powershell
git config --show-origin --get user.name
git config --show-origin --get user.email
```

正常情况下：

- 来源应指向当前项目的 `.git/config`；
- 邮箱应以 `@users.noreply.github.com` 结尾；
- 不应显示真实私人邮箱。

## 六、理解 `.gitignore` 的保护作用

`.gitignore` 告诉 Git 哪些内容不应该进入版本历史。本项目已经排除：

- `.venv/` 和第三方软件；
- Python 缓存；
- 原始 JSON 和 nuisance cache；
- 日志、临时文件和 smoke 结果；
- 论文 PDF 和导师汇报 Word；
- 模型权重；
- Release ZIP；
- `.env`、密钥和证书。

检查几个关键路径：

```powershell
git check-ignore -v .venv
git check-ignore -v results/raw
git check-ignore -v _release
git check-ignore -v 2602.11139v1.pdf
```

### 如何读输出

例如：

```text
.gitignore:2:.venv/    .venv
```

意思是 `.gitignore` 第 2 行的 `.venv/` 规则排除了该目录。

确认精简结果没有被忽略：

```powershell
git status --short results/published
```

正常情况下会看到 `results/published/` 中的候选文件，而不是完全没有输出。

## 七、提交前查看所有候选文件

先查看 Git 状态：

```powershell
git status --short
```

第一次提交前，很多行会以 `??` 开头。例如：

```text
?? README.md
?? src/
?? results/published/
```

`??` 的意思是 untracked：Git 看到了这些文件，但它们还没有进入暂存区。

只列出没有被忽略的候选路径：

```powershell
git ls-files --others --exclude-standard
```

计算候选数量：

```powershell
(git ls-files --others --exclude-standard | Measure-Object).Count
```

归档整理完成时的检查结果为 131 个候选文件。如果之后修改文档，文件数量一般不变；如果你新增了文件，数量可能增加，需要确认新增内容是什么。

### 候选内容应该包括

- `.gitignore`；
- `README.md`、`REPRODUCIBILITY.md`、`RESULTS.md`；
- `ARCHIVE_MANIFEST.md` 和本教程；
- `pyproject.toml`；
- `configs/`、`src/`、`scripts/`、`tests/`、`docs/`；
- `results/published/`。

### 候选内容不应该包括

- `.venv/`；
- `_release/`；
- `results/raw/`、`results/logs/` 或任何 smoke 目录；
- `.pdf`、`.docx` 或 `.zip`；
- 模型权重；
- 密码、Token 或私钥。

此时不要使用 `git add .`。虽然 `.gitignore` 已经提供保护，第一次学习时仍建议明确指定路径，便于理解自己正在加入什么。

## 八、把指定文件放入暂存区

执行：

```powershell
git add .gitignore README.md REPRODUCIBILITY.md RESULTS.md ARCHIVE_MANIFEST.md UPLOAD_GUIDE.md pyproject.toml configs src scripts tests docs results/published
```

### 这条命令做了什么

它把列出的文件和目录加入暂存区，但：

- 没有创建 commit；
- 没有连接 GitHub；
- 没有上传文件；
- 不会加入被 `.gitignore` 排除的内容。

查看暂存状态：

```powershell
git status
```

正常情况下会看到：

```text
Changes to be committed:
  new file:   README.md
  new file:   ...
```

`Changes to be committed` 表示这些文件已经装进“下一次 commit 的箱子”。

## 九、提交前进行第二次安全检查

### 1. 查看全部暂存文件名

```powershell
git diff --cached --name-only
```

`--cached` 表示检查暂存区，而不是检查普通工作区。

### 2. 查看文件数量和改动规模

```powershell
git diff --cached --stat
```

### 3. 自动检查禁传路径

```powershell
git diff --cached --name-only | Select-String -Pattern '(^|/)\.venv|(^|/)_release|^results/(?!published/)|\.pdf$|\.docx$|\.zip$|\.pt$|\.pth$'
```

正常结果是没有任何输出。没有输出表示暂存区没有匹配这些危险路径。

### 4. 检查常见密钥形式

```powershell
git grep --cached -n -I -E "github_pat_|ghp_|BEGIN .* PRIVATE KEY"
```

正常结果同样是没有输出。

### 如果发现放错文件

撤回整个暂存区：

```powershell
git restore --staged .
```

它只会把文件从暂存区拿出来，不会删除电脑上的实验文件。撤回后修改 `.gitignore` 或 `git add` 路径，再重新暂存。

如果只想撤回一个文件：

```powershell
git restore --staged "文件路径"
```

## 十、创建本地第一次 commit

安全检查通过后执行：

```powershell
git commit -m "Archive TabICLv2-DML experiments through Stage 3B"
```

### 提交说明是什么意思

- `git commit`：创建本地版本快照；
- `-m`：直接提供一行提交说明；
- 后面的英文说明表示“归档截至 Stage 3B 的实验”。

这一步仍然没有上传到 GitHub。

查看刚创建的提交：

```powershell
git log --oneline -1
git show --stat --oneline HEAD
```

你会看到一个类似 `a1b2c3d` 的短编号。它是 commit 的唯一标识前缀。

### 检查作者邮箱是否正确

```powershell
git show -s --format="Author: %an <%ae>%nCommit: %h %s" HEAD
```

邮箱必须是 `@users.noreply.github.com`。

如果还没 push 就发现作者邮箱错误，先设置正确邮箱，再修改最近一次 commit 作者：

```powershell
git config --local user.email "$noreplyEmail"
git commit --amend --reset-author --no-edit
```

再次运行 `git show -s` 检查。`--no-edit` 表示保持原提交说明不变。

## 十一、创建私有 GitHub 仓库并上传

确认本地 commit 正确后执行：

```powershell
gh repo create tabiclv2-dml-experiments --private --source . --remote origin --push
```

### 每个参数的意思

- `gh repo create`：在 GitHub 创建仓库；
- `tabiclv2-dml-experiments`：远程仓库名；
- `--private`：创建私有仓库；
- `--source .`：使用当前目录作为源项目；
- `--remote origin`：将远程仓库在本地命名为 `origin`；
- `--push`：创建后立即上传当前 `main` 分支。

`origin` 只是 Git 对主要远程仓库的惯用简称，不是 GitHub 用户名。

正常情况下，终端会显示仓库地址并报告 push 成功。

检查本地保存的远程地址：

```powershell
git remote -v
```

应看到 `origin` 的 fetch 和 push 地址。

检查 GitHub 仓库属性：

```powershell
gh repo view --json nameWithOwner,visibility,defaultBranchRef
```

重点确认：

- `visibility` 为 `PRIVATE`；
- 默认分支名称为 `main`；
- `nameWithOwner` 是你的 GitHub 用户名和仓库名。

在浏览器打开：

```powershell
gh repo view --web
```

### 如果同名仓库已经在网站上创建

如果 GitHub 已经存在 `tabiclv2-dml-experiments`，不要重复运行 `gh repo create`。执行：

```powershell
git remote add origin "https://github.com/$gitHubUser/tabiclv2-dml-experiments.git"
git push -u origin main
```

如果 `origin` 已经存在，先查看而不要覆盖：

```powershell
git remote -v
```

不要在不理解远程状态时使用强制推送。

## 十二、理解为什么原始结果使用 Release

普通 Git 适合代码和小型文本文件，不适合频繁记录上万个小 JSON 和可重新生成的缓存。

因此本项目分成两层：

```text
普通 Git 仓库：代码、配置、测试、说明、汇总 CSV、图表
GitHub Release：完整原始 JSON、必要的 NPZ nuisance cache、文件哈希清单
```

本机 Release 文件是：

```text
_release\tabiclv2-dml-experiments-v0.1-stage3b-raw-results.zip
```

该 ZIP 被 `.gitignore` 排除，但 GitHub CLI 仍然可以把它作为 Release 附件上传。这不是矛盾：不进入 Git commit，不代表不能作为独立附件上传。

## 十三、创建版本标签

标签可以理解为给某个 commit 贴上一个固定版本名称。

创建本地标签：

```powershell
git tag -a v0.1-stage3b -m "Experiments completed through Stage 3B"
```

查看标签指向：

```powershell
git show --no-patch v0.1-stage3b
```

把标签推送到 GitHub：

```powershell
git push origin v0.1-stage3b
```

分支 push 和标签 push 是两件事。前面的 `gh repo create --push` 上传了 `main`；这里单独上传版本标签。

## 十四、上传 Release 原始结果

上传前再次计算本地 ZIP 的 SHA-256：

```powershell
Get-FileHash "_release\tabiclv2-dml-experiments-v0.1-stage3b-raw-results.zip" -Algorithm SHA256
```

必须得到：

```text
E5B94D51B71110A437433ACF72BE8FC720358A86A86B0667DDC47DF6080602B7
```

如果不同，停止上传并检查文件，不要修改 `ARCHIVE_MANIFEST.md` 来迎合新的哈希。

创建 Release 并上传 ZIP：

```powershell
gh release create v0.1-stage3b "_release\tabiclv2-dml-experiments-v0.1-stage3b-raw-results.zip" --title "Stage 3B experiment archive" --notes-file ARCHIVE_MANIFEST.md
```

### 命令解读

- `gh release create`：创建 GitHub Release；
- `v0.1-stage3b`：Release 对应的 Git 标签；
- ZIP 路径：作为附件上传的完整原始结果；
- `--title`：GitHub 页面显示的标题；
- `--notes-file`：使用归档清单作为 Release 说明。

检查 Release：

```powershell
gh release view v0.1-stage3b --json tagName,name,isDraft,isPrerelease,assets
gh release view v0.1-stage3b --web
```

重点确认：

- 标签是 `v0.1-stage3b`；
- 不是 Draft；
- 不是 Prerelease；
- assets 中存在 ZIP；
- ZIP 大小约为 14.98 MB。

## 十五、下载一次并重新校验

远程页面显示附件不等于附件一定完整。建议下载一次进行闭环验证。

创建被 Git 忽略的检查目录：

```powershell
New-Item -ItemType Directory -Force "_release\download-check"
```

从 GitHub Release 下载：

```powershell
gh release download v0.1-stage3b --pattern "tabiclv2-dml-experiments-v0.1-stage3b-raw-results.zip" --dir "_release\download-check"
```

计算下载文件哈希：

```powershell
Get-FileHash "_release\download-check\tabiclv2-dml-experiments-v0.1-stage3b-raw-results.zip" -Algorithm SHA256
```

结果仍必须是：

```text
E5B94D51B71110A437433ACF72BE8FC720358A86A86B0667DDC47DF6080602B7
```

这证明本地 ZIP、上传附件和下载附件是同一个文件。

## 十六、最终检查清单

在 GitHub 网页确认：

- 仓库显示 Private；
- README 中文正常显示；
- `src/`、`scripts/`、`tests/` 和 `configs/` 存在；
- `results/published/` 中有汇总表和图表；
- 没有 `.venv`、原始 JSON、日志、论文 PDF 和 Word 文件；
- `v0.1-stage3b` 标签存在；
- Release 存在且带有 ZIP；
- Release ZIP 下载后的 SHA-256 正确。

在 PowerShell 检查：

```powershell
git status
git log --oneline --decorate -3
git tag
git remote -v
```

正常的 `git status` 应显示：

```text
nothing to commit, working tree clean
```

被 `.gitignore` 排除的本机原始结果仍然存在，只是默认不会出现在普通 `git status` 中。

## 十七、以后如何更新仓库

以后修改代码或文档时，基本流程是：

```powershell
git status
git diff
git add "具体文件或目录"
git diff --cached
git commit -m "清楚说明本次修改"
git push
```

建议：

- 每个 commit 只做一项明确工作；
- 提交说明写“改了什么”，不要只写 `update`；
- push 前检查作者邮箱；
- push 前检查暂存区；
- 不要长期使用 `git add .` 代替思考；
- 不要提交模型权重、虚拟环境和凭据。

例如补充下一阶段实验方案：

```powershell
git add docs RESULTS.md
git commit -m "Document the next tree-DGP confirmation experiment"
git push
```

## 十八、常见问题与处理

### 1. `gh` 不是可识别的命令

关闭并重新打开整个 PowerShell 窗口，再运行：

```powershell
gh --version
```

如果仍然无法识别，可临时使用完整路径：

```powershell
& "$env:LOCALAPPDATA\Programs\GitHub CLI\bin\gh.exe" --version
```

### 2. `fatal: not a git repository`

说明当前目录不对：

```powershell
cd "C:\Users\Stern\Desktop\论文实验"
git status
```

### 3. Push 因私人邮箱被阻止

先检查最近 commit：

```powershell
git show -s --format="%an <%ae>" HEAD
```

如果显示真实邮箱，改为 noreply 邮箱并重写最近一次尚未成功 push 的 commit：

```powershell
git config --local user.email "$noreplyEmail"
git commit --amend --reset-author --no-edit
git push
```

不要通过关闭 GitHub 的邮箱保护来绕过问题。

### 4. `repository already exists`

说明 GitHub 上已有同名仓库。先检查：

```powershell
gh repo view "$gitHubUser/tabiclv2-dml-experiments"
git remote -v
```

如果远程仓库就是你想使用的私有空仓库，再添加 `origin` 并 push。不要重复创建。

### 5. `remote origin already exists`

先查看现有地址：

```powershell
git remote get-url origin
```

如果地址正确，不需要再次 `git remote add`。如果地址不认识，停止操作并核查，不要直接覆盖或强制推送。

### 6. Release 已经存在

检查现有 Release：

```powershell
gh release view v0.1-stage3b
```

如果 Release 已创建但附件缺失，可补传：

```powershell
gh release upload v0.1-stage3b "_release\tabiclv2-dml-experiments-v0.1-stage3b-raw-results.zip"
```

如果附件同名但内容不确定，先在网页核查，不要盲目使用覆盖参数。

### 7. GitHub 提示文件太大

停止 push，执行：

```powershell
git diff --cached --stat
git diff --cached --name-only
```

检查是否误暂存 `.venv`、模型权重、ZIP、PDF 或原始结果。若发现，使用 `git restore --staged` 撤回。

### 8. Push 被拒绝或远程存在未知提交

不要立刻使用 `--force`。先检查：

```powershell
git status
git remote -v
git log --oneline --decorate --all -10
```

远程历史与本地不一致时，应先理解远程有什么内容，再决定合并或重建空仓库。第一次上传不需要强制推送。

## 十九、最短命令速查

只有在完整阅读并理解前面步骤后，才使用这份速查表：

```powershell
cd "C:\Users\Stern\Desktop\论文实验"
gh auth login
$gitHubUser = gh api user --jq .login
$gitHubId = gh api user --jq .id
$noreplyEmail = "$gitHubId+$gitHubUser@users.noreply.github.com"
git config --local user.name "$gitHubUser"
git config --local user.email "$noreplyEmail"
git add .gitignore README.md REPRODUCIBILITY.md RESULTS.md ARCHIVE_MANIFEST.md UPLOAD_GUIDE.md pyproject.toml configs src scripts tests docs results/published
git diff --cached --name-only
git commit -m "Archive TabICLv2-DML experiments through Stage 3B"
git show -s --format="Author: %an <%ae>%nCommit: %h %s" HEAD
gh repo create tabiclv2-dml-experiments --private --source . --remote origin --push
git tag -a v0.1-stage3b -m "Experiments completed through Stage 3B"
git push origin v0.1-stage3b
gh release create v0.1-stage3b "_release\tabiclv2-dml-experiments-v0.1-stage3b-raw-results.zip" --title "Stage 3B experiment archive" --notes-file ARCHIVE_MANIFEST.md
gh release view v0.1-stage3b --web
```

第一次操作时仍建议逐节执行，而不是直接粘贴这组命令。
