import type { Backend } from "../backend-registry/types";
import { getActiveBackend } from "../backend-registry/active-store";
import { callCloudProxy } from "./proxy";

export type LicenseState =
  "disabled" | "missing_key" | "pending" | "valid" | "invalid" | "unverified";

export interface LicenseStatus {
  state: LicenseState;
  /** Why the license is not valid, when `state` is `invalid`. */
  reason: string | null;
  /** Whether the install uses an offline license file. */
  offline: boolean;
  /** trial, paid or development. */
  type: string | null;
  /** Null when the license never expires or the date isn't known. */
  license_ends_at: string | null;
  expires_at: string | null;
  last_checked_at: string | null;
  customer_center_url: string;
}

function getActiveCloudBackend(): Backend {
  const active = getActiveBackend().backend;
  if (active.kind !== "cloud") {
    throw new Error("Cloud license call requires a cloud backend.");
  }
  return active;
}

/**
 * Fetch `GET /api/license/status`: the license state an OpenHands Enterprise
 * instance shows in its license banner. An instance that predates the route
 * answers with its web app's HTML instead of JSON; that is `null` here.
 */
export async function fetchCloudLicenseStatus(): Promise<LicenseStatus | null> {
  const backend = getActiveCloudBackend();
  const data = await callCloudProxy<LicenseStatus | string>({
    backend,
    method: "GET",
    path: "/api/license/status",
  });
  return typeof data === "object" &&
    data !== null &&
    typeof data.state === "string"
    ? data
    : null;
}
