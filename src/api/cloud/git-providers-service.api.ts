import { isProvider, type Provider } from "#/types/settings";
import { getActiveBackend } from "../backend-registry/active-store";
import type { Backend } from "../backend-registry/types";
import { callCloudProxy } from "./proxy";

/** A git provider the instance defines, and whether this user has linked it. */
export interface CloudGitProvider {
  /** `auth_providers.id`: what a conversation names as `git_provider_id`. */
  id: number;
  title: string;
  providerType: Provider;
  connected: boolean;
  /** The provider rejected the linked token; only re-linking fixes it. */
  needsReconnect: boolean;
}

interface GitProviderResponse {
  id: number;
  title: string;
  provider_type: string;
  connected: boolean;
  needs_reconnect: boolean;
}

function getActiveCloudBackend(): Backend {
  const active = getActiveBackend().backend;
  if (active.kind !== "cloud") {
    throw new Error("Cloud git providers call requires a cloud backend.");
  }
  return active;
}

/**
 * The git providers of an instance with `enable_oauth_v2_login`. This is the
 * list the instance's own Integrations settings page shows. Rows whose type
 * Canvas does not know are dropped.
 */
export async function listCloudGitProviders(): Promise<CloudGitProvider[]> {
  const rows = await callCloudProxy<GitProviderResponse[]>({
    backend: getActiveCloudBackend(),
    method: "GET",
    path: "/oauth/git-providers",
  });
  return rows.flatMap((row) =>
    isProvider(row.provider_type)
      ? [
          {
            id: row.id,
            title: row.title,
            providerType: row.provider_type,
            connected: row.connected,
            needsReconnect: row.needs_reconnect,
          },
        ]
      : [],
  );
}
