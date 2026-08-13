import { getSettingsSections } from "#/settings/registry";
import { useSettingsContext } from "#/settings/use-settings-context";
// Registers the SDK-schema settings sections (Condenser, Agent Context).
import "#/settings/register-sdk-settings-sections";

const CONDENSER_SETTINGS_PAGE = "/settings/condenser";

/**
 * Host for the Condenser settings page. Renders whatever sections are
 * registered for this page and visible in the current
 * {@link useSettingsContext} — the same host every settings page now uses.
 */
function CondenserSettingsScreen() {
  const context = useSettingsContext();
  const sections = getSettingsSections(CONDENSER_SETTINGS_PAGE, context);

  return (
    <div className="flex flex-col gap-6">
      {sections.map(({ id, Component }) => (
        <Component key={id} />
      ))}
    </div>
  );
}

export default CondenserSettingsScreen;
