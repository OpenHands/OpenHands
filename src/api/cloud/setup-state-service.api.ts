import type { Backend } from "../backend-registry/types";
import { callCloudProxy } from "./proxy";
import type { CloudSetupState } from "./types";

/**
 * Fetch `GET /api/admin/setup-state`: the Super Admin setup guide's
 * organization, dismissal and server-derived step progress.
 *
 * Enterprise backends only mount the route while the Super Admin feature is
 * on; otherwise the path falls through to the web app and the response is
 * not JSON. Anything that is not an object is treated as "no guide".
 */
export async function getCloudSetupState(
  backend: Backend,
): Promise<CloudSetupState> {
  const data = await callCloudProxy<Partial<CloudSetupState> | string>({
    backend,
    method: "GET",
    path: "/api/admin/setup-state",
  });
  const state = typeof data === "object" && data !== null ? data : {};
  return {
    guide_org_id: state.guide_org_id ?? null,
    guide_dismissed: state.guide_dismissed ?? false,
    guide_steps: state.guide_steps ?? null,
  };
}
