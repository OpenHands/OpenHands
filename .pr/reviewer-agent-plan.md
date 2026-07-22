# Reviewer Agent — 实现计划

## 概述

在 OpenHands 项目中构建一个 **自动化 PR Reviewer Agent**，使其在 PR 提交时自动触发多维度的代码审查（安全检查、代码风格、性能评估、PR 模板合规），并将审查结果以评论形式发布到 PR 上。

本计划基于 OpenHands 现有的基础设施设计：
- 已有 skills 系统（`.agents/skills/` + `skills/`）
- 已有 GitHub Actions 工作流
- 已有 `/codereview` 技能但仅支持手动触发
- 已有 `qa-changes-by-openhands.yml`（自动化 QA，可作为 reviewer 的参考模式）

---

## 阶段 0：环境准备与基础调研（1-2 天）

### 0.1 Fork 仓库并搭建开发环境

```bash
# Fork OpenHands/OpenHands 到你自己的 GitHub 账号
# 然后 clone
git clone https://github.com/<你的账号>/OpenHands.git
cd OpenHands

# 创建开发分支
git checkout -b feat/reviewer-agent

# 安装 pre-commit hooks（项目要求）
make install-pre-commit-hooks

# 安装前端依赖（如果涉及前端改动）
cd frontend && npm install
```

### 0.2 理解现有系统架构

需要阅读的关键文件：

| 文件 | 目的 |
|------|------|
| `skills/code-review.md` | 现有的代码审查 skill（手动触发） |
| `skills/codereview-roasted.md` | 另一种风格的审查 skill |
| `.agents/skills/custom-codereview-guide.md` | 仓库专属的审查指南 |
| `skills/README.md` | Skill 系统的架构说明 |
| `.github/workflows/pr.yml` | PR 标题 lint 工作流 |
| `.github/workflows/pr-readiness-confirm.yml` | PR 就绪确认工作流 |
| `.github/workflows/qa-changes-by-openhands.yml` | 自动化 QA 工作流（重要参考！） |
| `.github/pull_request_template.md` | PR 模板 |
| `AGENTS.md` | Agent 的指令文件 |

### 0.3 输出：调研报告

形成一份文档记录：
- 现有 code review 能力的边界（只能手动触发，无自动化）
- GitHub Actions 的触发模式（`pull_request_target` / `pull_request`）
- `.agents/skills/` 与 `skills/` 的关系
- 可以复用的基础设施（comment API、label 系统等）

---

## 阶段 1：Reviewer Agent 技能文件设计（2-3 天）

### 1.1 设计 Skill 文件结构

在 `.agents/skills/` 下创建 reviewer 技能，这是给 **OpenHands Agent 本身**使用的技能定义：

```
.agents/skills/reviewer/
├── SKILL.md              # 主技能文件
└── references/
    ├── review-rules.md   # 审查规则详细说明
    └── severity.md       # 严重级别定义
```

**SKILL.md 的核心设计：**

```yaml
---
name: reviewer
description: >
  自动化 PR 代码审查 Agent。当 PR 被创建或更新时触发，
  覆盖安全审查、代码风格、性能评估、PR 模板合规四个维度。
  支持中英文双语代码库审查。
triggers:
- /review
- review this PR
- 审查这个 PR
- run review
---

# Reviewer Agent

## 角色
你是一个自动化 PR 审查 Agent，集成在 CI/CD 流程中。
当 PR 被打开或更新时自动触发，以评论形式发布审查结果。

## 审查维度

### 1. PR 模板合规（前置检查）
- 检查 PR 描述是否包含模板要求的字段
- 缺失关键字段则标记为 BLOCKER

### 2. 安全审查（Security）
- 硬编码密钥/凭据
- SQL 注入风险
- XSS 风险
- 路径遍历
- CSRF 保护缺失

### 3. 代码质量与风格（Quality）
- 函数长度 > 50 行
- 文件长度 > 800 行
- 嵌套深度 > 4 层
- 未使用的 import/变量
- 不合适的命名
- 缺少错误处理

### 4. 性能评估（Performance）
- N+1 查询模式
- 缺少分页
- 不必要的重复计算
- 低效的数据结构选择

### 5. 测试覆盖（可选，取决于 PR 规模）
- 新功能是否包含测试
- Bug 修复是否包含回归测试

## 输出格式

审查结果以结构化评论发布到 PR 上：

```
## 🤖 Reviewer Agent Report

### PR Template Compliance
- ✅ / ❌ Why field present
- ✅ / ❌ Summary present
- ✅ / ❌ How to Test present

### 🔴 Critical
- [file:line] 描述

### 🟡 High
- [file:line] 描述

### 🔵 Medium
- [file:line] 描述

### Summary
- Total issues found: N
- Critical: N | High: N | Medium: N
- Overall: ✅ Approve / ⚠️ Changes Requested
```
```

### 1.2 设计参考文件

**references/review-rules.md** — 存放 OpenHands 项目特有的审查规则，参考现有 `custom-codereview-guide.md` 中的指南：
- i18n key 必须来自 `I18nKey` 枚举，禁止动态拼接
- 前端数据获取必须走 TanStack Query，禁止直接调用 API client
- 后端 Python 代码遵循 PEP 8
- Docker 镜像标签规范

**references/severity.md** — 定义严重级别判断标准：

```
CRITICAL: 安全漏洞、潜在数据丢失、硬编码密钥
HIGH: 功能 Bug、严重违反项目规范
MEDIUM: 可维护性问题、代码异味
LOW: 风格建议、可选改进
```

### 1.3 测试 Skill 文件

通过 OpenHands 的 `/review` 命令手动触发测试，验证：
- 技能被正确加载
- 审查维度覆盖完整
- 输出格式正确

---

## 阶段 2：GitHub Actions 自动化集成（3-4 天）

### 2.1 创建自动审查工作流

新建 `.github/workflows/reviewer.yml`：

```yaml
name: Reviewer Agent

on:
  pull_request:
    types: [opened, synchronize, ready_for_review]
  pull_request_review:
    types: [submitted]

jobs:
  reviewer:
    if: |
      github.event.pull_request.draft == false &&
      (
        github.event_name != 'pull_request_review' ||
        github.event.review.state == 'changes_requested'
      )
    runs-on: ubuntu-24.04
    permissions:
      contents: read
      pull-requests: write
      issues: write
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # 获取完整 git 历史用于 diff

      - name: Get PR diff
        id: diff
        run: |
          git diff origin/${{ github.base_ref }}...HEAD > /tmp/pr_diff.txt
          echo "diff_size=$(wc -c < /tmp/pr_diff.txt)" >> $GITHUB_OUTPUT

      - name: Run Reviewer Agent
        uses: OpenHands/extensions/plugins/qa-changes@main  # 参考现有模式
        with:
          llm-model: litellm_proxy/openai/gpt-4o
          llm-base-url: ${{ secrets.LLM_BASE_URL }}
          max-budget: '5.0'
          github-token: ${{ secrets.GITHUB_TOKEN }}
          # 传递 diff 和 PR 元数据供 Agent 分析
```

**设计决策说明：**

为什么参考 `qa-changes-by-openhands.yml` 而不是自己写 Python 脚本？

- OpenHands 项目已经在使用 `OpenHands/extensions/plugins/qa-changes@main` 这个 action
- 复用现有基础设施，降低维护成本
- 如果需要更大的灵活性，可以先用 Python 脚本实现核心逻辑（调用 LLM API + GitHub API）

**备选方案（更独立、更可控）：**

如果插件方式受限，可以用 Python 脚本直接实现：

```
.github/workflows/reviewer.yml  → 触发
scripts/reviewer/
├── main.py              # 入口：解析参数，编排审查流程
├── pr_analyzer.py       # 获取 PR 信息、diff、文件列表
├── review_engine.py     # 调用 LLM 进行审查
├── comment_builder.py   # 格式化审查结果为评论
├── severity.py           # 严重级别判断
└── rules/
    ├── security.py      # 安全规则
    ├── quality.py       # 代码质量规则
    └── performance.py   # 性能规则
```

### 2.2 实现审查逻辑（Python 方案）

**pr_analyzer.py** — 使用 `PyGithub` 或 GitHub API：

```python
from github import Github
import os

class PRAnalyzer:
    def __init__(self, token, repo_name, pr_number):
        self.client = Github(token)
        self.repo = self.client.get_repo(repo_name)
        self.pr = self.repo.get_pull(pr_number)

    def get_diff(self):
        """获取 PR 的完整 diff"""
        return self.pr.get_files()

    def get_pr_metadata(self):
        """获取 PR 元数据"""
        return {
            'title': self.pr.title,
            'body': self.pr.body,
            'author': self.pr.user.login,
            'base': self.pr.base.ref,
            'head': self.pr.head.ref,
            'changed_files': self.pr.changed_files,
            'additions': self.pr.additions,
            'deletions': self.pr.deletions,
        }

    def check_template_compliance(self):
        """检查 PR 模板合规"""
        required = ['Why', 'Summary', 'How to Test', 'Type']
        body = self.pr.body or ''
        missing = [s for s in required if s not in body]
        return missing
```

**review_engine.py** — 调用 LLM 进行审查：

```python
import openai
import json

class ReviewEngine:
    def __init__(self, api_key, model="gpt-4o"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def review_diff(self, diff_text, pr_metadata, rules):
        """对 diff 进行多维度审查"""
        prompt = self._build_prompt(diff_text, pr_metadata, rules)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是 OpenHands 项目的代码审查 Agent。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def _build_prompt(self, diff_text, pr_metadata, rules):
        return f"""
        请审查以下 PR 变更。

        PR 元数据:
        - 标题: {pr_metadata['title']}
        - 变更文件数: {pr_metadata['changed_files']}
        - 新增行: {pr_metadata['additions']}
        - 删除行: {pr_metadata['deletions']}

        Diff:
        ```
        {diff_text[:8000]}
        ```

        审查规则:
        {json.dumps(rules, ensure_ascii=False)}

        请按以下 JSON 格式输出:
        {{
            "template_compliance": {{"passed": bool, "missing": [str]}},
            "issues": [
                {{
                    "severity": "critical|high|medium|low",
                    "file": "path/to/file",
                    "line": int,
                    "category": "security|quality|performance",
                    "title": "问题简述",
                    "description": "详细描述",
                    "suggestion": "修改建议"
                }}
            ],
            "summary": {{
                "total": int,
                "critical": int,
                "high": int,
                "medium": int,
                "verdict": "approve|changes_requested"
            }}
        }}
        """
```

### 2.3 工作流设计决策

需要你决定的几个关键选择：

1. **触发时机**：
   - 每次 push 都审查？→ 可能过于频繁，消耗 API 额度
   - 推荐：仅在 PR **从 draft 转为 ready** 时 + **添加 `review-this` label** 时触发
   - 参考 `qa-changes-by-openhands.yml` 的 label 触发模式

2. **审查深度**：
   - 小型 PR（< 200 行 diff）：全量审查
   - 大型 PR（> 200 行 diff）：仅审查新增文件的关键部分，或分批审查

3. **LLM 模型选择**：
   - GPT-4o：质量优先（推荐用于 Reviewer）
   - Haiku 4.5：快速且经济（用于模板合规检查这种简单任务）

### 2.4 性能优化

- **增量审查**：只审查本次 commit 新增/修改的部分，而非全量
- **缓存**：对未修改的文件跳过审查
- **并发控制**：一个 PR 同时只允许一个审查运行（用 `concurrency` 控制）

---

## 阶段 3：与中国市场/中文字段相关的差异化能力（2 天）

### 3.1 双语审查支持

在 reviewer 中加入对中文代码场景的支持：

```
审查规则扩展:
- 中英文混排规范性检查（中文与英文之间是否有空格）
- 中文注释是否清晰（避免拼音注释）
- 中文 commit message 规范
- 项目中英文术语一致性（如 "用户" vs "user" 的混用）
```

### 3.2 输出示例

审查结果中增加中文友好的输出：

```
## 🤖 Reviewer Agent 审查报告

### 模板合规性
- ✅ Why 字段存在
- ✅ Summary 字段存在
- ❌ How to Test 字段缺失

### 🔴 严重问题
| 文件 | 行号 | 类别 | 问题 |
|------|------|------|------|
| src/auth.py | 42 | security | API key 硬编码 |
| frontend/api.ts | 15 | security | SQL 注入风险 |

### 🟡 高优先级
...

### 📝 中文规范检查
- `docs/guide.md:88` — 中英文之间缺少空格
- `src/utils.py:23` — 注释使用了拼音 'yonghu'，建议用中文 '用户'
```

### 3.3 面试价值分析

这个差异化能力在面试时非常有用，你可以说：

> "考虑到国内很多中厂代码库是中英文混合的，我特意在 Reviewer Agent 中加入了中文注释规范检查和中英文混排检查。这个能力在实际落地时很受国内团队欢迎。"

---

## 阶段 4：测试与验证（2 天）

### 4.1 单元测试

```python
# tests/reviewer/test_severity.py
def test_classify_hardcoded_secret():
    assert classify_issue("password = 'sk-xxx'") == Severity.CRITICAL

def test_classify_long_function():
    assert classify_issue("function with 60 lines") == Severity.MEDIUM
```

### 4.2 在真实 PR 上测试

1. 创建一个测试 PR（包含各类代码问题）
2. 观察 reviewer 是否能正确识别
3. 调整 prompt 和规则直到输出稳定

### 4.3 对比测试

用同样的 PR 分别运行：
- 手动 `/codereview` → 记录输出
- Reviewer Agent → 记录输出

对比覆盖率和准确率。

---

## 阶段 5：文档与面试准备（1-2 天）

### 5.1 项目文档

- 在 `skills/README.md` 中添加 reviewer 技能的说明
- 编写 Reviewer Agent 的使用指南

### 5.2 面试叙事

准备以下几段话：

**项目背景：**
"OpenHands 是一个 7.8 万行代码的开源 AI 编程助手平台，社区活跃，PR 数量多。之前代码审查全靠人工手动触发 `/codereview`，效率低、覆盖不全。我设计了一个自动化的 Reviewer Agent 来解决这个问题。"

**技术方案：**
"我基于 OpenHands 现有的 skill 系统和 GitHub Actions 搭建了自动化审查 pipeline。核心是一个多维度审查引擎——安全、质量、性能、模板合规四维覆盖。用 GPT-4o 做深度审查，Haiku 做快速模板检查，权衡了质量和成本。"

**差异化亮点：**
"我还加入了中英文双语审查支持，能检查中英文混排规范、中文注释质量等。这个设计特别适合国内团队的混合语言代码库。"

**量化成果：**
"覆盖了 X 个审查规则维度，在测试 PR 上达到了 Y% 的问题检出率，审查时间从人工的 Z 分钟缩短到自动的 N 分钟。"

---

## 时间线总结

| 阶段 | 内容 | 预计时间 |
|------|------|----------|
| 0 | 环境搭建与调研 | 1-2 天 |
| 1 | Skill 文件设计 | 2-3 天 |
| 2 | GitHub Actions 自动化集成 | 3-4 天 |
| 3 | 中文场景差异化能力 | 2 天 |
| 4 | 测试与验证 | 2 天 |
| 5 | 文档与面试准备 | 1-2 天 |
| **合计** | | **11-15 天** |

> 如果你想缩短周期，可以：
> - 阶段 3 合并到阶段 1 和 2 中（不单独做，而是把中文规则融入审查引擎）
> - 阶段 2 优先实现 Python 脚本方案（更可控）

---

## 关键决策点（已确认）

| 决策 | 选择 | 理由 |
|------|------|------|
| 实现路径 | **Python 脚本** | 不受插件限制，更灵活可控 |
| 触发方式 | **label 触发**（`review-this`） | 避免每次 push 消耗 API，参考 `qa-this` 模式 |
| 提交策略 | **先在 fork 验证** | 验证好了再考虑提 PR 给上游 |
