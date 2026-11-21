import { getRegisteredBackends } from "../backend-registry/active-store";
import { callCloudProxy } from "./proxy";

export interface SandboxTemplateOption {
  id: string;
  display_name: string;
}

export interface SandboxSelectionOptions {
  enabled: boolean;
  items: SandboxTemplateOption[];
}

export const SANDBOX_SELECTION_PATH = "/api/v1/sandbox-specs/selection";

export async function getSandboxSelectionOptions(
  backendId: string,
  orgId: string,
): Promise<SandboxSelectionOptions> {
  const backend = getRegisteredBackends().find((item) => item.id === backendId);
  if (backend?.kind !== "cloud") {
    throw new Error("Sandbox templates require a registered cloud backend.");
  }
  const raw = await callCloudProxy<Partial<SandboxSelectionOptions>>({
    backend,
    method: "GET",
    path: SANDBOX_SELECTION_PATH,
    orgId,
  });
  // The selection endpoint is best-effort metadata for the launcher UI. A
  // response missing `items`, or shipping a non-array `items`, must degrade
  // to "no sandbox picker" instead of crashing the launcher downstream.
  if (!Array.isArray(raw?.items)) {
    return { enabled: false, items: [] };
  }
  return { enabled: raw.enabled === true, items: raw.items };
}
