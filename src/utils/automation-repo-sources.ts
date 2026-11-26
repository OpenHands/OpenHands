import type { AutomationRepository } from "#/types/automation";
import type { Provider } from "#/types/settings";
import { getGitProviderBaseUrl } from "#/utils/utils";
import { parseGitRemoteUrl } from "#/utils/parse-git-remote-url";

/** A `repos` entry in a preset create or update request. */
export type AutomationRepoSource = {
  url: string;
  ref?: string;
  provider?: string;
};

// The setup picker lists GitHub repositories by their `owner/repo` shorthand,
// and the service needs a provider to clone a shorthand.
const SHORTHAND_PROVIDER = "github";
const FULL_GIT_URL_PATTERN = /^(?:https?:\/\/|git@)/i;

/** Providers whose repositories a run clones when the service does not say. */
export const DEFAULT_CLONE_PROVIDERS: Provider[] = [
  "github",
  "gitlab",
  "bitbucket",
];

/**
 * The providers whose repositories the automation service's runs clone, from
 * its `repoProviders` capability. An older service without the field clones
 * from the three hosted providers. Only these names are valid as a `repos`
 * entry's `provider`; the service rejects any other.
 */
export function automationCloneProviders(
  repoProviders?: string[] | null,
): Provider[] {
  return repoProviders
    ? (repoProviders as Provider[])
    : DEFAULT_CLONE_PROVIDERS;
}

/**
 * The base URL a connected provider is served on: scheme, host, port and any
 * path prefix, without a trailing slash. A self-managed instance keeps its
 * port (`https://git.example.com:8443`) and prefix (`https://example.com/gitlab`).
 */
export function providerBaseUrl(
  host: string | null | undefined,
): string | null {
  const trimmed = host?.trim();
  if (!trimmed) return null;
  try {
    const url = new URL(
      trimmed.includes("://") ? trimmed : `https://${trimmed}`,
    );
    if (!url.hostname) return null;
    return `${url.origin}${url.pathname.replace(/\/+$/, "")}`;
  } catch {
    return null;
  }
}

/** The base URL of the provider's hosted service, or null when it has none. */
function hostedBaseUrl(provider: Provider): string | null {
  return providerBaseUrl(getGitProviderBaseUrl(provider));
}

// One provider name covers the hosted service and self-managed instances alike
// for GitLab and Forgejo, so a connection whose host the server does not
// report cannot be addressed: the hosted URL would silently point a
// self-managed repository at the public service. Bitbucket Data Center has no
// hosted service at all. Bitbucket names its self-managed product separately,
// and OpenHands' Azure DevOps integration serves dev.azure.com only, so a
// missing host there is the hosted service.
const HOST_REQUIRED_PROVIDERS: Provider[] = [
  "gitlab",
  "forgejo",
  "bitbucket_data_center",
];

/**
 * Whether a connected provider's listed repositories can be addressed: GitHub
 * by its shorthand, the others on the reported host or on their hosted
 * service, unless the name also covers self-managed instances and no host was
 * reported.
 */
export function providerHostKnown(
  provider: Provider,
  host?: string | null,
): boolean {
  if (providerBaseUrl(host)) return true;
  return provider === "github" || !HOST_REQUIRED_PROVIDERS.includes(provider);
}

/**
 * The repository field entry for a repository picked from a provider, or null
 * when the connection's host is unknown (see `providerHostKnown`).
 *
 * GitHub keeps its `owner/repo` shorthand. Other providers, and a GitHub
 * Enterprise host, get the full URL on the connected host or on the provider's
 * hosted service, which the service recognizes and clones with its token.
 */
export function automationRepositoryAddress(
  provider: Provider,
  fullName: string,
  host?: string | null,
): string | null {
  const configured = providerBaseUrl(host);
  if (provider === "github" && !configured) return fullName;
  if (!providerHostKnown(provider, host)) return null;
  if (provider === "azure_devops") {
    // Azure DevOps names a repository organization/project/repository, always
    // on dev.azure.com: the server stores an Azure DevOps connection's
    // organization (sometimes as a URL with a path) in its `host`, not an
    // instance, so that value is never a base URL.
    const [organization, project, ...rest] = fullName.split("/");
    const repository = rest.join("/");
    if (!organization || !project || !repository) return fullName;
    const path = [organization, project].map(encodeURIComponent).join("/");
    return `${hostedBaseUrl(provider)}/${path}/_git/${encodeURIComponent(repository)}`;
  }
  if (provider === "bitbucket_data_center" && configured) {
    // Data Center clones a repository from /scm/PROJECT/repo.git; a personal
    // repository's project is `~user`.
    const [project, ...rest] = fullName.split("/");
    const repository = rest.join("/");
    if (!project || !repository) return fullName;
    const path = [project, repository].map(encodeURIComponent).join("/");
    return `${configured}/scm/${path}.git`;
  }
  const base = configured ?? hostedBaseUrl(provider);
  if (!base) return fullName;
  return `${base}/${fullName}`;
}

/**
 * The provider a full URL belongs to when it points at a connected
 * self-hosted instance, matched on the same base URL the address was built
 * from. A hosted service's URL needs no provider: the service recognizes
 * those hosts itself. Only a provider the run can clone is named; a host of
 * any other connected provider gets none, since the service rejects the name.
 */
export function providerForAddress(
  url: string,
  hosts: Partial<Record<Provider, string | null>> = {},
  cloneProviders: Provider[] = DEFAULT_CLONE_PROVIDERS,
): Provider | undefined {
  let address: string;
  try {
    const parsed = new URL(url);
    address = `${parsed.origin}${parsed.pathname}`;
  } catch {
    return undefined;
  }
  // Two connected instances can share an origin with one base nested under
  // the other (`https://example.com` and `https://example.com/gitlab`), so the
  // longest matching base wins rather than whichever key comes first.
  let match: { provider: Provider; base: string } | undefined;
  for (const [provider, host] of Object.entries(hosts) as [
    Provider,
    string | null,
  ][]) {
    // An Azure DevOps connection's `host` is its organization, not a base URL.
    if (!cloneProviders.includes(provider) || provider === "azure_devops") {
      continue;
    }
    const base = providerBaseUrl(host);
    if (
      base &&
      base !== hostedBaseUrl(provider) &&
      (address === base || address.startsWith(`${base}/`)) &&
      (!match || base.length > match.base.length)
    ) {
      match = { provider, base };
    }
  }
  if (match) return match.provider;
  return undefined;
}

/**
 * The `repos` a request sends for the setup form's repositories.
 *
 * A repository the automation already clones keeps its saved branch and
 * provider. A new shorthand is a GitHub repository. A full URL is sent with a
 * provider only when it points at a connected self-hosted host, so the
 * service authenticates its clone for that host and for no other.
 */
export function buildAutomationRepoSources(
  repositories: string[],
  saved: AutomationRepository[] = [],
  hosts: Partial<Record<Provider, string | null>> = {},
  cloneProviders: Provider[] = DEFAULT_CLONE_PROVIDERS,
): AutomationRepoSource[] {
  return repositories.map((url) => {
    const existing = saved.find((repo) => repo.url === url);
    if (existing) {
      return {
        url: existing.url,
        ...(existing.ref && { ref: existing.ref }),
        ...(existing.provider && { provider: existing.provider }),
      };
    }
    if (!FULL_GIT_URL_PATTERN.test(url)) {
      return { url, provider: SHORTHAND_PROVIDER };
    }
    const provider = providerForAddress(url, hosts, cloneProviders);
    return provider ? { url, provider } : { url };
  });
}

/**
 * How the automation pages name a saved repository: its `owner/repo` path, and
 * for Bitbucket Data Center the project and repository without the `/scm/`
 * clone prefix (or the context path before it).
 */
export function automationRepositoryLabel(
  repository: AutomationRepository,
): string {
  const path = parseGitRemoteUrl(repository.url)?.repository ?? repository.url;
  return repository.provider === "bitbucket_data_center"
    ? path.replace(/^(?:.*\/)?scm\//, "")
    : path;
}

/** Whether the form lists other repositories than the automation clones. */
export function automationRepositoriesChanged(
  repositories: string[],
  saved: AutomationRepository[] = [],
): boolean {
  return (
    repositories.length !== saved.length ||
    repositories.some((url, index) => url !== saved[index]?.url)
  );
}

/** Repositories in the setup form's comma-separated shape. */
export function formatAutomationRepositories(
  repositories: AutomationRepository[] = [],
): string {
  return repositories
    .map((repo) => repo.url.trim())
    .filter(Boolean)
    .join(", ");
}
