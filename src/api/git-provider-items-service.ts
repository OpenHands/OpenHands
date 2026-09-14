import { SettingsClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { SecretsService } from "#/api/secrets-service";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import type { Provider } from "#/types/settings";
import { constructPullRequestUrl, getGitProviderBaseUrl } from "#/utils/utils";

export interface GitProviderItem {
  id: number;
  number: number;
  title: string;
  url: string;
  authorLogin: string | null;
  updatedAt: string | null;
}

const PROVIDER_TOKEN_SECRET_CANDIDATES: Partial<Record<Provider, string[]>> = {
  github: ["GITHUB_TOKEN", "GH_TOKEN", "github"],
  gitlab: ["GITLAB_TOKEN", "GL_TOKEN", "gitlab"],
  bitbucket: ["BITBUCKET_TOKEN", "bitbucket"],
  forgejo: ["FORGEJO_TOKEN", "forgejo"],
};

const LIST_LIMIT = 30;

async function resolveProviderToken(
  provider: Provider,
): Promise<string | null> {
  // Cloud backends keep provider tokens server-side; browser-side secret
  // lookup is only meaningful for the local agent-server secrets store.
  if (getActiveBackend().backend.kind !== "local") {
    return null;
  }

  const candidates = PROVIDER_TOKEN_SECRET_CANDIDATES[provider] ?? [];
  if (candidates.length === 0) {
    return null;
  }

  const secrets = await SecretsService.getSecrets();
  const available = new Set(secrets.map((secret) => secret.name));
  const match = candidates.find((name) => available.has(name));
  if (!match) {
    return null;
  }

  try {
    return await new SettingsClient(getAgentServerClientOptions()).getSecret(
      match,
    );
  } catch {
    return null;
  }
}

function constructIssueUrl(
  issueNumber: number,
  provider: Provider,
  repositoryName: string,
): string {
  const baseUrl = getGitProviderBaseUrl(provider);
  switch (provider) {
    case "gitlab":
      return `${baseUrl}/${repositoryName}/-/issues/${issueNumber}`;
    case "bitbucket":
      return `${baseUrl}/${repositoryName}/issues/${issueNumber}`;
    case "forgejo":
      return `${baseUrl}/${repositoryName}/issues/${issueNumber}`;
    case "github":
    default:
      return `${baseUrl}/${repositoryName}/issues/${issueNumber}`;
  }
}

function constructIssuesListUrl(
  provider: Provider,
  repositoryName: string,
): string {
  const baseUrl = getGitProviderBaseUrl(provider);
  switch (provider) {
    case "gitlab":
      return `${baseUrl}/${repositoryName}/-/issues`;
    case "bitbucket":
      return `${baseUrl}/${repositoryName}/issues`;
    case "forgejo":
      return `${baseUrl}/${repositoryName}/issues`;
    case "github":
    default:
      return `${baseUrl}/${repositoryName}/issues`;
  }
}

function constructPullRequestsListUrl(
  provider: Provider,
  repositoryName: string,
): string {
  const baseUrl = getGitProviderBaseUrl(provider);
  switch (provider) {
    case "gitlab":
      return `${baseUrl}/${repositoryName}/-/merge_requests`;
    case "bitbucket":
      return `${baseUrl}/${repositoryName}/pull-requests`;
    case "forgejo":
      return `${baseUrl}/${repositoryName}/pulls`;
    case "github":
    default:
      return `${baseUrl}/${repositoryName}/pulls`;
  }
}

async function fetchGithubJson<T>(
  path: string,
  token: string | null,
): Promise<T> {
  const headers: Record<string, string> = {
    Accept: "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const response = await fetch(`https://api.github.com${path}`, { headers });
  if (!response.ok) {
    throw new Error(`GitHub API ${response.status}`);
  }
  return response.json() as Promise<T>;
}

async function fetchGitlabJson<T>(
  path: string,
  token: string | null,
): Promise<T> {
  const headers: Record<string, string> = {
    Accept: "application/json",
  };
  if (token) {
    headers["PRIVATE-TOKEN"] = token;
  }

  // gitlab.com's public REST API — an external host, not the agent-server
  // `/api` surface the rule guards (its `/api/v4` prefix just trips the
  // substring heuristic; the companion no-direct-agent-server-calls test
  // matches only relative `/api/...` URLs and is not affected).
  // eslint-disable-next-line local/no-direct-agent-server-fetch
  const response = await fetch(`https://gitlab.com/api/v4${path}`, { headers });
  if (!response.ok) {
    throw new Error(`GitLab API ${response.status}`);
  }
  return response.json() as Promise<T>;
}

type GithubIssueOrPr = {
  id: number;
  number: number;
  title: string;
  html_url: string;
  user?: { login?: string } | null;
  updated_at?: string | null;
  pull_request?: unknown;
};

type GitlabMergeRequest = {
  id: number;
  iid: number;
  title: string;
  web_url: string;
  author?: { username?: string } | null;
  updated_at?: string | null;
};

type GitlabIssue = {
  id: number;
  iid: number;
  title: string;
  web_url: string;
  author?: { username?: string } | null;
  updated_at?: string | null;
};

/**
 * Lists open pull requests / merge requests and issues for a connected
 * repository by calling the provider's public REST API from the browser.
 * When a matching token secret exists locally (e.g. `GITHUB_TOKEN`), it is
 * used for auth so private repos work; otherwise public-repo unauthenticated
 * requests are attempted.
 */
export class GitProviderItemsService {
  static constructIssuesListUrl = constructIssuesListUrl;

  static constructPullRequestsListUrl = constructPullRequestsListUrl;

  static async listPullRequests(
    repository: string,
    provider: Provider,
  ): Promise<GitProviderItem[]> {
    const token = await resolveProviderToken(provider);

    if (provider === "gitlab") {
      const encoded = encodeURIComponent(repository);
      const items = await fetchGitlabJson<GitlabMergeRequest[]>(
        `/projects/${encoded}/merge_requests?state=opened&per_page=${LIST_LIMIT}`,
        token,
      );
      return items.map((item) => ({
        id: item.id,
        number: item.iid,
        title: item.title,
        url: item.web_url,
        authorLogin: item.author?.username ?? null,
        updatedAt: item.updated_at ?? null,
      }));
    }

    if (provider !== "github" && provider !== "forgejo") {
      return [];
    }

    const [owner, repo] = repository.split("/");
    if (!owner || !repo) {
      return [];
    }

    const items = await fetchGithubJson<GithubIssueOrPr[]>(
      `/repos/${owner}/${repo}/pulls?state=open&per_page=${LIST_LIMIT}`,
      token,
    );

    return items.map((item) => ({
      id: item.id,
      number: item.number,
      title: item.title,
      url:
        item.html_url ||
        constructPullRequestUrl(item.number, provider, repository),
      authorLogin: item.user?.login ?? null,
      updatedAt: item.updated_at ?? null,
    }));
  }

  static async listIssues(
    repository: string,
    provider: Provider,
  ): Promise<GitProviderItem[]> {
    const token = await resolveProviderToken(provider);

    if (provider === "gitlab") {
      const encoded = encodeURIComponent(repository);
      const items = await fetchGitlabJson<GitlabIssue[]>(
        `/projects/${encoded}/issues?state=opened&per_page=${LIST_LIMIT}`,
        token,
      );
      return items.map((item) => ({
        id: item.id,
        number: item.iid,
        title: item.title,
        url: item.web_url,
        authorLogin: item.author?.username ?? null,
        updatedAt: item.updated_at ?? null,
      }));
    }

    if (provider !== "github" && provider !== "forgejo") {
      return [];
    }

    const [owner, repo] = repository.split("/");
    if (!owner || !repo) {
      return [];
    }

    const items = await fetchGithubJson<GithubIssueOrPr[]>(
      `/repos/${owner}/${repo}/issues?state=open&per_page=${LIST_LIMIT}`,
      token,
    );

    // GitHub's issues endpoint also returns pull requests.
    return items
      .filter((item) => !item.pull_request)
      .map((item) => ({
        id: item.id,
        number: item.number,
        title: item.title,
        url:
          item.html_url || constructIssueUrl(item.number, provider, repository),
        authorLogin: item.user?.login ?? null,
        updatedAt: item.updated_at ?? null,
      }));
  }
}
