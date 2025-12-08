# Instructions for developing SAAS locally

You have a few options here, which are expanded on below:

- A simple local development setup, with live reloading for both OSS and this repo
- A more complex setup that includes Redis
- An even more complex setup that includes GitHub events

## Prerequisites

Before starting, make sure you have the following tools installed:

### Required for all options:

- [gcloud CLI](https://cloud.google.com/sdk/docs/install) - For authentication and secrets management
- [sops](https://github.com/mozilla/sops) - For secrets decryption
  - macOS: `brew install sops`
  - Linux: `sudo apt-get install sops` or download from GitHub releases
  - Windows: Install via Chocolatey `choco install sops` or download from GitHub releases

### Additional requirements for enabling GitHub webhook events

- make
- Python development tools (build-essential, python3-dev)
- [ngrok](https://ngrok.com/download) - For creating tunnels to localhost

## Option 1: Simple local development

This option will allow you to modify the both the OSS code and the code in this repo,
and see the changes in real-time.

This option works best for most scenarios. The only thing it's missing is
the GitHub events webhook, which is not necessary for most development.

### 1. OpenHands location

The open source OpenHands repo should be cloned as a sibling directory,
in `../OpenHands`. This is hard-coded in the pyproject.toml (edit if necessary)

If you're doing this the first time, you may need to run

```
poetry update openhands-ai
```

### 2. Set up env

First run this to retrieve Github App secrets

```
gcloud auth application-default login
gcloud config set project global-432717
local/decrypt_env.sh
```

Now run this to generate a `.env` file, which will used to run SAAS locally

```
python -m pip install PyYAML
export LITE_LLM_API_KEY=<your LLM API key>
python enterprise_local/convert_to_env.py
```

You'll also need to set up the runtime image, so that the dev server doesn't try to rebuild it.

```
export SANDBOX_RUNTIME_CONTAINER_IMAGE=ghcr.io/openhands/runtime:main-nikolaik
docker pull $SANDBOX_RUNTIME_CONTAINER_IMAGE
```

By default the application will log in json, you can override.

```
export LOG_PLAIN_TEXT=1
```

### 3. Start the OpenHands frontend

Start the frontend like you normally would in the open source OpenHands repo.

### 4. Start the SaaS backend

```
make build

make start-backend
```

You should have a server running on `localhost:3000`, similar to the open source backend.
Oauth should work properly.

## Option 2: With Redis

Follow all the steps above, then setup redis:

```bash
docker run  -p 6379:6379 --name openhands-redis -d redis
export REDIS_HOST=host.docker.internal # you may want this to be localhost
export REDIS_PORT=6379
```

## Option 3: Work with GitHub events

### 1. Setup env file

(see above)

### 2. Build OSS Openhands

Develop on [Openhands](https://github.com/All-Hands-AI/OpenHands) locally. When ready, run the following inside Openhands repo (not the Deploy repo)

```
docker build -f containers/app/Dockerfile -t openhands .
```

### 3. Build SAAS Openhands

Build the SAAS image locally inside Deploy repo. Note that `openhands` is the name of the image built in Step 2

```
docker build -t openhands-saas ./app/ --build-arg BASE="openhands"
```

### 4. Create a tunnel

Run in a separate terminal

```
ngrok http 3000
```

There will be a line

```
Forwarding                    https://bc71-2603-7000-5000-1575-e4a6-697b-589e-5801.ngrok-free.app
```

Remember this URL as it will be used in Step 5 and 6

### 5. Setup Staging Github App callback/webhook urls

Using the URL found in Step 4, add another callback URL (`https://bc71-2603-7000-5000-1575-e4a6-697b-589e-5801.ngrok-free.app/oauth/github/callback`)

### 6. Run

This is the last step! Run SAAS openhands locally using

```
docker run --env-file ./app/.env -p 3000:3000 openhands-saas
```

Note `--env-file` is what injects the `.env` file created in Step 1

Visit the tunnel domain found in Step 4 to run the app (`https://bc71-2603-7000-5000-1575-e4a6-697b-589e-5801.ngrok-free.app`)

### Local Debugging with VSCode

Local Development necessitates running a version of OpenHands that is as similar as possible to the version running in the SAAS Environment. Before running these steps, it is assumed you have a local development version of the OSS OpenHands project running.

#### Redis

A Local redis instance is required for clustered communication between server nodes. The standard docker instance will suffice.
`docker run -it -p 6379:6379 --name my-redis -d redis`

#### Postgres

A Local postgres instance is required. I used the official docker image:
`docker run -p 5432:5432 --name my-postgres -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=openhands -d postgres`
Run the alembic migrations:
`poetry run alembic upgrade head `

#### VSCode launch.json

The VSCode launch.json below sets up 2 servers to test clustering, running independently on localhost:3030 and localhost:3031. Running only the server on 3030 is usually sufficient unless tests of the clustered functionality are required. Secrets may be harvested directly from staging by connecting...
`kubectl exec --stdin --tty <POD_NAME> -n <NAMESPACE> -- /bin/bash`
And then invoking `printenv`. NOTE: _DO NOT DO THIS WITH PROD!!!_ (Hopefully by the time you read this, nobody will have access.)

```
{
    "configurations": [
        {
            "name": "Python Debugger: Python File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}"
        },
        {
            "name": "OpenHands Deploy",
            "type": "debugpy",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "saas_server:app",
                "--reload",
                "--host",
                "0.0.0.0",
                "--port",
                "3030"
            ],
            "env": {
                "DEBUG": "1",
                "FILE_STORE": "local",
                "REDIS_HOST": "localhost:6379",
                "OPENHANDS": "<YOUR LOCAL OSS OPENHANDS DIR>",
                "FRONTEND_DIRECTORY": "<YOUR LOCAL OSS OPENHANDS DIR>/frontend/build",
                "SANDBOX_RUNTIME_CONTAINER_IMAGE": "ghcr.io/openhands/runtime:main-nikolaik",
                "FILE_STORE_PATH": "<YOUR HOME DIRECTORY>>/.openhands-state",
                "OPENHANDS_CONFIG_CLS": "server.config.SaaSServerConfig",
                "GITHUB_APP_ID": "1062351",
                "GITHUB_APP_PRIVATE_KEY": "<GITHUB PRIVATE KEY>",
                "GITHUB_APP_CLIENT_ID": "Iv23lis7eUWDQHIq8US0",
                "GITHUB_APP_CLIENT_SECRET": "<GITHUB CLIENT SECRET>",
                "POSTHOG_CLIENT_KEY": "<POSTHOG CLIENT KEY>",
                "LITE_LLM_API_URL": "https://llm-proxy.staging.all-hands.dev",
                "LITE_LLM_TEAM_ID": "62ea39c4-8886-44f3-b7ce-07ed4fe42d2c",
                "LITE_LLM_API_KEY": "<LITE LLM API KEY>"
            },
            "justMyCode": false,
            "cwd": "${workspaceFolder}/app"
        },
        {
            "name": "OpenHands Deploy 2",
            "type": "debugpy",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "saas_server:app",
                "--reload",
                "--host",
                "0.0.0.0",
                "--port",
                "3031"
            ],
            "env": {
                "DEBUG": "1",
                "FILE_STORE": "local",
                "REDIS_HOST": "localhost:6379",
                "OPENHANDS": "<YOUR LOCAL OSS OPENHANDS DIR>",
                "FRONTEND_DIRECTORY": "<YOUR LOCAL OSS OPENHANDS DIR>/frontend/build",
                "SANDBOX_RUNTIME_CONTAINER_IMAGE": "ghcr.io/openhands/runtime:main-nikolaik",
                "FILE_STORE_PATH": "<YOUR HOME DIRECTORY>>/.openhands-state",
                "OPENHANDS_CONFIG_CLS": "server.config.SaaSServerConfig",
                "GITHUB_APP_ID": "1062351",
                "GITHUB_APP_PRIVATE_KEY": "<GITHUB PRIVATE KEY>",
                "GITHUB_APP_CLIENT_ID": "Iv23lis7eUWDQHIq8US0",
                "GITHUB_APP_CLIENT_SECRET": "<GITHUB CLIENT SECRET>",
                "POSTHOG_CLIENT_KEY": "<POSTHOG CLIENT KEY>",
                "LITE_LLM_API_URL": "https://llm-proxy.staging.all-hands.dev",
                "LITE_LLM_TEAM_ID": "62ea39c4-8886-44f3-b7ce-07ed4fe42d2c",
                "LITE_LLM_API_KEY": "<LITE LLM API KEY>"
            },
            "justMyCode": false,
            "cwd": "${workspaceFolder}/app"
        },
        {
            "name": "Unit Tests",
            "type": "debugpy",
            "request": "launch",
            "module": "pytest",
            "args": [
                "./tests/unit",
                //"./tests/unit/test_clustered_conversation_manager.py",
                "--durations=0"
            ],
            "env": {
                "DEBUG": "1"
            },
            "justMyCode": false,
            "cwd": "${workspaceFolder}/app"
        },
        // set working directory...
    ]
}
```
