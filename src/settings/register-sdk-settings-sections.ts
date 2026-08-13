import { registerSettingsSection } from "./registry";
import { CondenserSection } from "#/components/features/settings/condenser-settings/condenser-section";
import { AgentContextSection } from "#/components/features/settings/agent-context-settings/agent-context-section";

/**
 * Register the SDK-schema-driven settings pages that are a single section each
 * (Condenser, Agent Context) as registry sections, so their routes use the
 * same host as the Application page instead of rendering a bespoke body.
 *
 * These are first-party sections registered from OSS code; importing this
 * module for its side effect (see the corresponding routes) performs the
 * registration, which is idempotent by section id.
 */
export function registerSdkSettingsSections(): void {
  registerSettingsSection({
    id: "condenser.main",
    page: "/settings/condenser",
    order: 10,
    Component: CondenserSection,
  });

  registerSettingsSection({
    id: "agent-context.main",
    page: "/settings/agent-context",
    order: 10,
    Component: AgentContextSection,
  });
}

registerSdkSettingsSections();
