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
  return callCloudProxy<SandboxSelectionOptions>({
    backend,
    method: "GET",
    path: SANDBOX_SELECTION_PATH,
    orgId,
  });
}
