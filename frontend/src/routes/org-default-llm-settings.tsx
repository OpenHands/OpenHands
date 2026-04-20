import { createPermissionGuard } from "#/utils/org/permission-guard";
import { LlmSettingsScreen } from "./llm-settings";

export const clientLoader = createPermissionGuard(
  "view_llm_settings",
  "/settings",
);

function OrgDefaultLlmSettingsScreen() {
  return <LlmSettingsScreen scope="org" />;
}

export default OrgDefaultLlmSettingsScreen;
