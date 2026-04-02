# How to use E2B

[E2B](https://e2b.dev) is an [open-source](https://github.com/e2b-dev/e2b) secure cloud environment (sandbox) made for running AI-generated code and agents. E2B offers [Python](https://pypi.org/project/e2b/) and [JS/TS](https://www.npmjs.com/package/e2b) SDK to spawn and control these sandboxes.

## Getting started

1. [Get your API key](https://e2b.dev/docs/getting-started/api-key)

1. Set your E2B API key to the `E2B_API_KEY` env var when starting the Docker container

1. **Optional** - Install the CLI with NPM.
    ```sh
    npm install -g @e2b/cli@latest
    ```
    Full CLI API is [here](https://e2b.dev/docs/cli/installation).

## OpenHands sandbox
You can use the E2B CLI to create a custom sandbox with a Dockerfile. Read the full guide [here](https://e2b.dev/docs/guide/custom-sandbox). The premade OpenHands sandbox for E2B is set up in the [`third_party/containers/e2b-sandbox` directory](../../../containers/e2b-sandbox), and it's called `openhands`.

## Debugging
You can connect to a running E2B sandbox with E2B CLI in your terminal.

- List all running sandboxes (based on your API key)
    ```sh
    e2b sandbox list
    ```

- Connect to a running sandbox
    ```sh
    e2b sandbox connect <sandbox-id>
    ```

## Links
- [E2B Docs](https://e2b.dev/docs)
- [E2B GitHub](https://github.com/e2b-dev/e2b)
