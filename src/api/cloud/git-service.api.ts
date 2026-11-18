import type {
  BranchPage,
  GitRepository,
  InstallationPage,
  RepositoryPage,
} from "#/types/git";
import type { Provider } from "#/types/settings";
import { getActiveBackend } from "../backend-registry/active-store";
import type { Backend } from "../backend-registry/types";
import { callCloudProxy } from "./proxy";

/** Names the exact connected git provider; omitted when none is chosen. */
function setGitProviderId(
  params: URLSearchParams,
  gitProviderId: number | null | undefined,
) {
  if (gitProviderId != null) {
    params.set("git_provider_id", String(gitProviderId));
  }
}

function getActiveCloudBackend(): Backend {
  const active = getActiveBackend().backend;
  if (active.kind !== "cloud") {
    throw new Error("Cloud git call requires a cloud backend.");
  }
  return active;
}

export async function searchCloudRepositories(args: {
  provider: Provider;
  query?: string;
  limit?: number;
  pageId?: string;
  installationId?: string;
  gitProviderId?: number | null;
}): Promise<RepositoryPage> {
  const backend = getActiveCloudBackend();
  const params = new URLSearchParams();
  params.set("provider", args.provider);
  params.set("limit", String(args.limit ?? 100));
  if (args.query) params.set("query", args.query);
  if (args.pageId) params.set("page_id", args.pageId);
  if (args.installationId) params.set("installation_id", args.installationId);
  setGitProviderId(params, args.gitProviderId);

  const data = await callCloudProxy<{
    items: GitRepository[];
    next_page_id: string | null;
  }>({
    backend,
    method: "GET",
    path: `/api/v1/git/repositories/search?${params.toString()}`,
  });

  return {
    items: data?.items ?? [],
    next_page_id: data?.next_page_id ?? null,
  };
}

/**
 * One repository with its default branch (`main_branch`). Repository lists
 * leave that field empty for some providers (e.g. Bitbucket Data Center).
 */
export async function getCloudRepositoryDetails(args: {
  provider: Provider;
  repository: string;
  gitProviderId?: number | null;
}): Promise<GitRepository> {
  const backend = getActiveCloudBackend();
  const params = new URLSearchParams();
  params.set("provider", args.provider);
  params.set("repository", args.repository);
  setGitProviderId(params, args.gitProviderId);

  return callCloudProxy<GitRepository>({
    backend,
    method: "GET",
    path: `/api/v1/git/repositories/details?${params.toString()}`,
  });
}

export async function getCloudInstallations(args: {
  provider: Provider;
  pageId?: string;
  limit?: number;
  gitProviderId?: number | null;
}): Promise<InstallationPage> {
  const backend = getActiveCloudBackend();
  const params = new URLSearchParams();
  params.set("provider", args.provider);
  params.set("limit", String(args.limit ?? 100));
  if (args.pageId) params.set("page_id", args.pageId);
  setGitProviderId(params, args.gitProviderId);

  const data = await callCloudProxy<{
    items: string[];
    next_page_id: string | null;
  }>({
    backend,
    method: "GET",
    path: `/api/v1/git/installations/search?${params.toString()}`,
  });

  return {
    items: data?.items ?? [],
    next_page_id: data?.next_page_id ?? null,
  };
}

export async function getCloudRepositoryBranches(args: {
  provider: Provider;
  repository: string;
  query?: string;
  pageId?: string;
  limit?: number;
  gitProviderId?: number | null;
}): Promise<BranchPage> {
  const backend = getActiveCloudBackend();
  const params = new URLSearchParams();
  params.set("provider", args.provider);
  params.set("repository", args.repository);
  params.set("limit", String(args.limit ?? 30));
  params.set("query", args.query ?? "");
  if (args.pageId) params.set("page_id", args.pageId);
  setGitProviderId(params, args.gitProviderId);

  const data = await callCloudProxy<{
    items: BranchPage["items"];
    next_page_id: string | null;
  }>({
    backend,
    method: "GET",
    path: `/api/v1/git/branches/search?${params.toString()}`,
  });

  return {
    items: data?.items ?? [],
    next_page_id: data?.next_page_id ?? null,
  };
}
