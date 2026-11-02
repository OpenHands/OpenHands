import type { Backend } from "#/api/backend-registry/types";

/**
 * The cloud instance's Settings > Integrations page. `org` is consumed by the
 * cloud settings loader so the page opens on the org that is active here
 * instead of the cloud's last-used org.
 */
export function cloudIntegrationsUrl(
  backend: Pick<Backend, "host">,
  orgId?: string | null,
): string {
  const orgQuery = orgId ? `?org=${encodeURIComponent(orgId)}` : "";
  return `${backend.host.replace(/\/+$/, "")}/settings/integrations${orgQuery}`;
}
