import { getActiveBackend } from "../backend-registry/active-store";
import type { Backend } from "../backend-registry/types";
import type { CustomSecretWithoutValue } from "../secrets-service.types";
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

export async function createCloudSecret(
  name: string,
  value: string,
  description?: string,
): Promise<void> {
  const backend = getActiveCloudBackend();
  await callCloudProxy<unknown>({
    backend,
    method: "POST",
    path: "/api/v1/secrets",
    body: { name, value, description },
  });
}

/**
 * Rename and/or redescribe an existing cloud secret. The cloud `PUT` endpoint
 * does not accept a value field — it only updates name + description — which
 * matches what `useUpdateSecret` actually sends from the secret-edit form.
 */
export async function updateCloudSecret(
  secretToEdit: string,
  name: string,
  description?: string,
): Promise<void> {
  const backend = getActiveCloudBackend();
  await callCloudProxy<unknown>({
    backend,
    method: "PUT",
    path: `/api/v1/secrets/${encodeURIComponent(secretToEdit)}`,
    body: { name, description },
  });
}

export async function deleteCloudSecret(name: string): Promise<void> {
  const backend = getActiveCloudBackend();
  await callCloudProxy<unknown>({
    backend,
    method: "DELETE",
    path: `/api/v1/secrets/${encodeURIComponent(name)}`,
  });
}
