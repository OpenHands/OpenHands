<a name="readme-top"></a>

<div align="center">
  <img src="https://raw.githubusercontent.com/All-Hands-AI/docs/main/openhands/static/img/logo.png" alt="Logo" width="200">
  <h1 align="center">OpenHands: Code Less, Make More</h1>
</div>


<div align="center">
  <a href="https://github.com/OpenHands/OpenHands/graphs/contributors"><img src="https://img.shields.io/github/contributors/OpenHands/OpenHands?style=for-the-badge&color=blue" alt="Contributors"></a>
  <a href="https://github.com/OpenHands/OpenHands/stargazers"><img src="https://img.shields.io/github/stars/OpenHands/OpenHands?style=for-the-badge&color=blue" alt="Stargazers"></a>
  <a href="https://github.com/OpenHands/OpenHands/blob/main/LICENSE"><img src="https://img.shields.io/github/license/OpenHands/OpenHands?style=for-the-badge&color=blue" alt="MIT License"></a>
  <br/>
  <a href="https://dub.sh/openhands"><img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join our Slack community"></a>
  <a href="https://github.com/OpenHands/OpenHands/blob/main/CREDITS.md"><img src="https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white" alt="Credits"></a>
  <br/>
  <a href="https://docs.all-hands.dev/usage/getting-started"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation"></a>
  <a href="https://arxiv.org/abs/2407.16741"><img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Paper on Arxiv"></a>
  <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0"><img src="https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Evaluation Benchmark Score"></a>

  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=de">Deutsch</a> |
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=es">Español</a> |
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=fr">français</a> |
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=ja">日本語</a> |
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=ko">한국어</a> |
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=pt">Português</a> |
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=ru">Русский</a> |
  <a href="https://www.readme-i18n.com/OpenHands/OpenHands?lang=zh">中文</a>

  <hr>
</div>

Welcome to OpenHands (formerly OpenDevin), a platform for software development agents powered by AI.

OpenHands agents can do anything a human developer can: modify code, run commands, browse the web,
call APIs, and yes—even copy code snippets from StackOverflow.

Learn more at [docs.all-hands.dev](https://docs.all-hands.dev), or [sign up for OpenHands Cloud](https://app.all-hands.dev) to get started.


> [!IMPORTANT]
> **Upcoming change**: We are renaming our GitHub Org from `All-Hands-AI` to `OpenHands` on October 20th, 2025.
> Check the [tracking issue](https://github.com/All-Hands-AI/OpenHands/issues/11376) for more information.


> [!IMPORTANT]
> Using OpenHands for work? We'd love to chat! Fill out
> [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform)
> to join our Design Partner program, where you'll get early access to commercial features and the opportunity to provide input on our product roadmap.

## ☁️ OpenHands Cloud
The easiest way to get started with OpenHands is on [OpenHands Cloud](https://app.all-hands.dev),
which comes with $10 in free credits for new users.

## 💻 Running OpenHands Locally

### Option 1: CLI Launcher (Recommended)

The easiest way to run OpenHands locally is using the CLI launcher with [uv](https://docs.astral.sh/uv/). This provides better isolation from your current project's virtual environment and is required for OpenHands' default MCP servers.

**Install uv** (if you haven't already):

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for the latest installation instructions for your platform.

**Launch OpenHands**:
```bash
# Launch the GUI server
uvx --python 3.12 openhands serve

# Or launch the CLI
uvx --python 3.12 openhands
```

You'll find OpenHands running at [http://localhost:3000](http://localhost:3000) (for GUI mode)!

### Option 2: Docker

<details>
<summary>Click to expand Docker command</summary>

You can also run OpenHands directly with Docker:

```bash
docker pull docker.openhands.dev/openhands/runtime:0.62-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.openhands.dev/openhands/runtime:0.62-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.openhands.dev/openhands/openhands:0.62
```

</details>

> **Note**: If you used OpenHands before version 0.44, you may want to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history to the new location.

> [!WARNING]
> On a public network? See our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation)
> to secure your deployment by restricting network binding and implementing additional security measures.

### Getting Started

When you open the application, you'll be asked to choose an LLM provider and add an API key.
[Anthropic's Claude Sonnet 4.5](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-5-20250929`)
works best, but you have [many options](https://docs.all-hands.dev/usage/llms).

See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for
system requirements and more information.

## 💡 Other ways to run OpenHands

> [!WARNING]
> OpenHands is meant to be run by a single user on their local workstation.
> It is not appropriate for multi-tenant deployments where multiple users share the same instance. There is no built-in authentication, isolation, or scalability.
>
> If you're interested in running OpenHands in a multi-tenant environment, check out the source-available, commercially-licensed
> [OpenHands Cloud Helm Chart](https://github.com/openHands/OpenHands-cloud)

You can [connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem),
interact with it via a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode),
run OpenHands in a scriptable [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode),
or run it on tagged issues with [a github action](https://docs.all-hands.dev/usage/how-to/github-action).

Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for more information and setup instructions.

If you want to modify the OpenHands source code, check out [Development.md](https://github.com/OpenHands/OpenHands/blob/main/Development.md).

Having issues? The [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting) can help.

## 📖 Documentation

To learn more about the project, and for tips on using OpenHands,
check out our [documentation](https://docs.all-hands.dev/usage/getting-started).

There you'll find resources on how to use different LLM providers,
troubleshooting resources, and advanced configuration options.

## 🤝 How to Join the Community

OpenHands is a community-driven project, and we welcome contributions from everyone. We do most of our communication
through Slack, so this is the best place to start, but we also are happy to have you contact us on Github:

- [Join our Slack workspace](https://dub.sh/openhands) - Here we talk about research, architecture, and future development.
- [Read or post Github Issues](https://github.com/OpenHands/OpenHands/issues) - Check out the issues we're working on, or add your own ideas.

See more about the community in [COMMUNITY.md](./COMMUNITY.md) or find details on contributing in [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Progress

See the monthly OpenHands roadmap [here](https://github.com/orgs/OpenHands/projects/1) (updated at the maintainer's meeting at the end of each month).

<p align="center">
  <a href="https://star-history.com/#OpenHands/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=OpenHands/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License, with the exception of the `enterprise/` folder. See [`LICENSE`](./LICENSE) for more information.

## 🙏 Acknowledgements

OpenHands is built by a large number of contributors, and every contribution is greatly appreciated! We also build upon other open source projects, and we are deeply thankful for their work.

For a list of open source projects and licenses used in OpenHands, please see our [CREDITS.md](./CREDITS.md) file.

## 📚 Cite

```
@inproceedings{
  wang2025openhands,
  title={OpenHands: An Open Platform for {AI} Software Developers as Generalist Agents},
  author={Xingyao Wang and Boxuan Li and Yufan Song and Frank F. Xu and Xiangru Tang and Mingchen Zhuge and Jiayi Pan and Yueqi Song and Bowen Li and Jaskirat Singh and Hoang H. Tran and Fuqiang Li and Ren Ma and Mingzhang Zheng and Bill Qian and Yanjun Shao and Niklas Muennighoff and Yizhe Zhang and Binyuan Hui and Junyang Lin and Robert Brennan and Hao Peng and Heng Ji and Graham Neubig},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=OJd3ayDDoF}
}
```
