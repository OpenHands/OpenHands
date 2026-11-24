import type { Backend } from "#/api/backend-registry/types";

/**
 * A page of the cloud instance's settings. `org` is consumed by the cloud
 * settings loader so the page opens on the org that is active here instead of
 * the cloud's last-used org.
 */
export function cloudSettingsUrl(
  backend: Pick<Backend, "host">,
  path: string,
  orgId?: string | null,
): string {
  const orgQuery = orgId ? `?org=${encodeURIComponent(orgId)}` : "";
  return `${backend.host.replace(/\/+$/, "")}${path}${orgQuery}`;
}

/** The cloud instance's Settings > Integrations page. */
export function cloudIntegrationsUrl(
  backend: Pick<Backend, "host">,
  orgId?: string | null,
): string {
  return cloudSettingsUrl(backend, "/settings/integrations", orgId);
}
