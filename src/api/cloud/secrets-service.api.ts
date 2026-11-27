import { getActiveBackend } from "../backend-registry/active-store";
import type { Backend } from "../backend-registry/types";
import type { CustomSecretWithoutValue } from "../secrets-service.types";
import { withRetry } from "../with-retry";
import { callCloudProxy } from "./proxy";

interface CloudSecretsPage {
  items: CustomSecretWithoutValue[];
  next_page_id: string | null;
}

const PAGE_LIMIT = 100;

function getActiveCloudBackend(): Backend {
  const active = getActiveBackend().backend;
  if (active.kind !== "cloud") {
    throw new Error("Cloud secrets call requires a cloud backend.");
  }
  return active;
}

/**
 * Walk every page of the cloud `/api/v1/secrets/search` endpoint and return
 * the merged list. The cloud shape (name + description) matches
 * `CustomSecretWithoutValue`, so items pass through unchanged.
 */
export async function fetchCloudSecrets(): Promise<CustomSecretWithoutValue[]> {
  const backend = getActiveCloudBackend();

  const secrets: CustomSecretWithoutValue[] = [];
  let pageId: string | null = null;

  do {
    const query = new URLSearchParams({ limit: String(PAGE_LIMIT) });
    if (pageId) query.set("page_id", pageId);

    const page = await callCloudProxy<CloudSecretsPage>({
      backend,
      method: "GET",
      path: `/api/v1/secrets/search?${query.toString()}`,
    });

    secrets.push(...(page.items ?? []));
    pageId = page.next_page_id;
  } while (pageId);

  return secrets;
}

export interface SaveCloudSecretOptions {
  /** Name the secret should have after saving. */
  name: string;
  /** New value. Omit to leave the stored value untouched. */
  value?: string;
  description?: string;
  /** Name the secret is currently stored under, when editing an existing one. */
  previousName?: string;
}

/**
 * Create a cloud secret, or save changes to an existing one.
 *
 * The cloud splits a save across two endpoints, so this issues up to two
 * requests to cover the whole operation:
 *
 * - `PUT /api/v1/secrets/{previousName}` applies the name and description. It
 *   is the only endpoint that can rename, and the only one that rejects a
 *   collision with `400`. It never touches the value.
 * - `POST /api/v1/secrets` writes the value. It is a documented upsert
 *   ("creates a new custom secret, or overwrites it if it already exists") but
 *   it is keyed by the name in its body, so it cannot rename, and its `value`
 *   field is required, so it cannot express a metadata-only edit.
 *
 * The `PUT` runs first so a rejected rename fails before the value is
 * overwritten — the cloud exposes no way to read a value back, so a value lost
 * to a half-applied save is unrecoverable. Each request is retried on its own:
 * re-running the `PUT` after a failed `POST` would `404` once the rename has
 * landed.
 */
export async function saveCloudSecret({
  name,
  value,
  description,
  previousName,
}: SaveCloudSecretOptions): Promise<void> {
  const backend = getActiveCloudBackend();

  if (previousName !== undefined) {
    await withRetry(() =>
      callCloudProxy<unknown>({
        backend,
        method: "PUT",
        path: `/api/v1/secrets/${encodeURIComponent(previousName)}`,
        body: { name, description },
      }),
    );
  }

  if (value !== undefined) {
    await withRetry(() =>
      callCloudProxy<unknown>({
        backend,
        method: "POST",
        path: "/api/v1/secrets",
        body: { name, value, description },
      }),
    );
  }
}

export async function deleteCloudSecret(name: string): Promise<void> {
  const backend = getActiveCloudBackend();
  await callCloudProxy<unknown>({
    backend,
    method: "DELETE",
    path: `/api/v1/secrets/${encodeURIComponent(name)}`,
  });
}
