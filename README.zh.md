<a name="readme-top"></a>

<div align="center">
  <img src="https://raw.githubusercontent.com/OpenHands/docs/main/openhands/static/img/logo.png" alt="Logo" width="200">
  <h1 align="center" style="border-bottom: none">OpenHands: AI 驱动的软件开发</h1>
</div>

<div align="center">
  <a href="https://github.com/OpenHands/OpenHands/blob/main/LICENSE"><img src="https://img.shields.io/badge/LICENSE-MIT-20B2AA?style=for-the-badge" alt="MIT License"></a>
  <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=811504672#gid=811504672"><img src="https://img.shields.io/badge/SWEBench-77.6-00cc00?logoColor=FFE165&style=for-the-badge" alt="Benchmark Score"></a>
  <br/>
  <a href="https://docs.openhands.dev/sdk"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="查看文档"></a>
  <a href="https://arxiv.org/abs/2511.03690"><img src="https://img.shields.io/badge/Paper-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="技术报告"></a>

  <!-- 保留这些链接。翻译会随着 README 自动更新。 -->
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=de">Deutsch</a> |
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=es">Español</a> |
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=fr">français</a> |
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=ja">日本語</a> |
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=ko">한국어</a> |
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=pt">Português</a> |
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=ru">Русский</a> |
  <a href="README.md">English</a>

</div>

<hr>

🙌 欢迎来到 OpenHands，这是一个专注于 AI 驱动开发的[社区](COMMUNITY.md)。我们非常欢迎您[加入我们的 Slack 频道](https://dub.sh/openhands)。

使用 OpenHands 有以下几种方式：

### OpenHands 软件智能体 SDK (Software Agent SDK)
SDK 是一个可组合的 Python 库，包含了我们所有的智能体技术。它是驱动以下所有功能的底层引擎。

在代码中定义智能体，然后可以在本地运行它们，或者在云端扩展至数千个智能体。

[查看文档](https://docs.openhands.dev/sdk) 或 [查看源码](https://github.com/OpenHands/software-agent-sdk/)

### OpenHands 命令行工具 (CLI)
CLI 是开始使用 OpenHands 最简单的方式。如果你使用过像 Claude Code 或 Codex 这样的工具，你会对这种体验感到非常熟悉。你可以使用 Claude、GPT 或任何其他大语言模型 (LLM) 来驱动它。

[查看文档](https://docs.openhands.dev/openhands/usage/run-openhands/cli-mode) 或 [查看源码](https://github.com/OpenHands/OpenHands-CLI)

### OpenHands 本地图形界面 (Local GUI)
使用本地 GUI 可以在你的笔记本电脑上运行智能体。它带有一个 REST API 和一个单页 React 应用程序。
如果你使用过 Devin 或 Jules，会对这种体验感到非常熟悉。

[查看文档](https://docs.openhands.dev/openhands/usage/run-openhands/local-setup) 或查看本仓库中的源码。

### OpenHands 云端版 (Cloud)
这是 OpenHands GUI 的部署版本，运行在托管基础设施上。

你可以通过[使用 GitHub 或 GitLab 账号登录](https://app.all-hands.dev)，免费使用 Minimax 模型进行体验。

OpenHands 云端版包含了公开源码的功能和集成：
- 与 Slack, Jira, 和 Linear 的集成
- 多用户支持
- 基于角色的访问控制 (RBAC) 和权限管理
- 协作功能（例如：对话分享）

### OpenHands 企业版 (Enterprise)
大型企业可以与我们合作，通过 Kubernetes 将 OpenHands Cloud 自托管在企业自己的虚拟私有云 (VPC) 中。
OpenHands 企业版也可以与上述的 CLI 和 SDK 协同工作。

OpenHands 企业版是公开源码的——你可以在此处的 `enterprise/` 目录中查看所有源代码，但如果你想运行它超过一个月，你需要购买许可证。

企业版合同还包含扩展支持和访问我们的研究团队的权限。

了解更多请访问 [openhands.dev/enterprise](https://openhands.dev/enterprise)

### 其他一切

查看我们的[产品路线图 (Product Roadmap)](https://github.com/orgs/openhands/projects/1)，如果您有任何想要看到的功能，请随时[提交 Issue](https://github.com/OpenHands/OpenHands/issues)！

您可能还对我们的[评估基础设施 (evaluation infrastructure)](https://github.com/OpenHands/benchmarks)、[Chrome 浏览器扩展 (chrome extension)](https://github.com/OpenHands/openhands-chrome-extension/)，或者我们的[心智理论模块 (Theory-of-Mind module)](https://github.com/OpenHands/ToM-SWE)感兴趣。

我们的所有工作都在 MIT 许可证下开源，但本仓库中的 `enterprise/` 目录除外（详情请参阅[企业版许可证](enterprise/LICENSE)）。
核心的 `openhands` 和 `agent-server` Docker 镜像也完全在 MIT 许可证下开源。

如果您需要任何帮助，或者只是想聊聊天，[快来 Slack 找我们吧](https://dub.sh/openhands)。