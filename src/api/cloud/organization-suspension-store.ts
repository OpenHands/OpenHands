import { HttpError } from "@openhands/typescript-client";

export type OrganizationSuspension = "organization" | "membership";

type Listener = () => void;

/**
 * The cloud backend's exact `detail` for a request made in a suspended
 * organization or with a suspended membership.
 */
const SUSPENSION_DETAILS: Record<string, OrganizationSuspension> = {
  "Organization is suspended": "organization",
  "User membership is suspended": "membership",
};

let suspensions: Record<string, OrganizationSuspension> = {};
const listeners = new Set<Listener>();

function suspensionKey(backendId: string, orgId: string): string {
  return `${backendId}:${orgId}`;
}

/**
 * Remember that the cloud backend refused `orgId` because it, or the user's
 * membership in it, is suspended. Any other error is ignored.
 */
export function recordOrganizationSuspension(
  backendId: string,
  orgId: string | null,
  error: unknown,
): void {
  if (!orgId || !(error instanceof HttpError) || error.status !== 403) return;
  const detail = (error.response as { detail?: unknown } | undefined)?.detail;
  const suspension =
    typeof detail === "string" ? SUSPENSION_DETAILS[detail] : undefined;
  if (!suspension) return;

  const key = suspensionKey(backendId, orgId);
  if (suspensions[key] === suspension) return;
  suspensions = { ...suspensions, [key]: suspension };
  listeners.forEach((listener) => listener());
}

export function getOrganizationSuspension(
  backendId: string,
  orgId: string | null,
): OrganizationSuspension | null {
  if (!orgId) return null;
  return suspensions[suspensionKey(backendId, orgId)] ?? null;
}

export function subscribeOrganizationSuspension(
  listener: Listener,
): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}
