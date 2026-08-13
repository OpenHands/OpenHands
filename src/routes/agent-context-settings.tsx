import { getSettingsSections } from "#/settings/registry";
import { useSettingsContext } from "#/settings/use-settings-context";
// Registers the SDK-schema settings sections (Condenser, Agent Context).
import "#/settings/register-sdk-settings-sections";

const AGENT_CONTEXT_SETTINGS_PAGE = "/settings/agent-context";

/**
 * Host for the Agent Context settings page. Renders whatever sections are
 * registered for this page and visible in the current
 * {@link useSettingsContext} — the same host every settings page now uses.
 */
function AgentContextSettingsScreen() {
  const context = useSettingsContext();
  const sections = getSettingsSections(AGENT_CONTEXT_SETTINGS_PAGE, context);

  return (
    <div className="flex flex-col gap-6">
      {sections.map(({ id, Component }) => (
        <Component key={id} />
      ))}
    </div>
  );
}

export default AgentContextSettingsScreen;
