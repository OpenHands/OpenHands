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
}): Promise<RepositoryPage> {
  const backend = getActiveCloudBackend();
  const params = new URLSearchParams();
  params.set("provider", args.provider);
  params.set("limit", String(args.limit ?? 100));
  if (args.query) params.set("query", args.query);
  if (args.pageId) params.set("page_id", args.pageId);
  if (args.installationId) params.set("installation_id", args.installationId);

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

export async function getCloudInstallations(args: {
  provider: Provider;
  pageId?: string;
  limit?: number;
}): Promise<InstallationPage> {
  const backend = getActiveCloudBackend();
  const params = new URLSearchParams();
  params.set("provider", args.provider);
  params.set("limit", String(args.limit ?? 100));
  if (args.pageId) params.set("page_id", args.pageId);

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
}): Promise<BranchPage> {
  const backend = getActiveCloudBackend();
  const params = new URLSearchParams();
  params.set("provider", args.provider);
  params.set("repository", args.repository);
  params.set("limit", String(args.limit ?? 30));
  params.set("query", args.query ?? "");
  if (args.pageId) params.set("page_id", args.pageId);

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
