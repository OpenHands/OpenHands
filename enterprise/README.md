# OpenHands Enterprise Server
> [!WARNING]
> This software is licensed under the [Polyform Free Trial License](./LICENSE). This is **NOT** an open source license. Usage is limited to 30 days per calendar year without a commercial license. If you would like to use it beyond 30 days, please [contact us](https://www.all-hands.dev/contact).

> [!WARNING]
> This is a work in progress and may contain bugs, incomplete features, or breaking changes.

This directory contains the enterprise server used by [OpenHands Cloud](https://github.com/All-Hands-AI/OpenHands-Cloud/). The official, public version of OpenHands Cloud is available at
[app.all-hands.dev](https://app.all-hands.dev).

You may also want to check out the MIT-licensed [OpenHands](https://github.com/OpenHands/OpenHands)

## Extension of OpenHands (OSS)

The code in `/enterprise` directory builds on top of open source (OSS) code, extending its functionality. The enterprise code is entangled with the OSS code in two ways

- Enterprise stacks on top of OSS. For example, the middleware in enterprise is stacked right on top of the middlewares in OSS. In `SAAS`, the middleware from BOTH repos will be present and running (which can sometimes cause conflicts)

- Enterprise overrides the implementation in OSS (only one is present at a time). For example, the server config SaasServerConfig which overrides [`ServerConfig`](https://github.com/OpenHands/OpenHands/blob/main/openhands/server/config/server_config.py#L8) on OSS. This is done through dynamic imports ([see here](https://github.com/OpenHands/OpenHands/blob/main/openhands/server/config/server_config.py#L37-#L45))

Key areas that change on `SAAS` are

- Authentication
- User settings
- etc

### Authentication

| Aspect                    | OSS                                                    | Enterprise                                                                                                                                 |
| ------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Authentication Method** | User adds a personal access token (PAT) through the UI | User performs OAuth through the UI. The Github app provides a short-lived access token and refresh token                            |
| **Token Storage**         | PAT is stored in **Settings**                          | Token is stored in **GithubTokenManager** (a file store in our backend)                                                             |
| **Authenticated status**  | We simply check if token exists in `Settings`          | We issue a signed cookie with `github_user_id` during oauth, so subsequent requests with the cookie can be considered authenticated |

Note that in the future, authentication will happen via keycloak. All modifications for authentication will happen in enterprise.

### GitHub Service

The github service is responsible for interacting with Github APIs. As a consequence, it uses the user's token and refreshes it if need be

| Aspect                    | OSS                                    | Enterprise                                            |
| ------------------------- | -------------------------------------- | ---------------------------------------------- |
| **Class used**            | `GitHubService`                        | `SaaSGitHubService`                            |
| **Token used**            | User's PAT fetched from `Settings`     | User's token fetched from `GitHubTokenManager` |
| **Refresh functionality** | **N/A**; user provides PAT for the app | Uses the `GitHubTokenManager` to refresh       |

NOTE: in the future we will simply replace the `GithubTokenManager` with keycloak. The `SaaSGithubService` should interact with keycloack instead.

# Areas that are BRITTLE!

## User ID vs User Token

- On OSS, the entire APP revolves around the Github token the user sets. `openhands/server` uses `request.state.github_token` for the entire app
- On Enterprise, the entire APP resolves around the Github User ID. This is because the cookie sets it, so `openhands/server` AND `enterprise/server` depend on it and completly ignore `request.state.github_token` (token is fetched from `GithubTokenManager` instead)

Note that introducing Github User ID on OSS, for instance, will cause large breakages.
