import { getSettingsSections } from "#/settings/registry";
import { useSettingsContext } from "#/settings/use-settings-context";
// Registers the built-in Application settings sections as a side effect.
import "#/settings/register-app-settings-sections";

const APP_SETTINGS_PAGE = "/settings/app";

/**
 * Host for the Application settings page. Rather than rendering a fixed
 * sequence of controls, it renders whatever sections are registered for this
 * page and visible in the current {@link useSettingsContext} — so
 * backend-specific (and, later, plugin-contributed) sections can be added by
 * registration instead of by editing this file. Each section owns its own
 * persistence.
 */
export function AppSettingsScreen() {
  const context = useSettingsContext();
  const sections = getSettingsSections(APP_SETTINGS_PAGE, context);

  return (
    <div data-testid="app-settings-screen" className="flex flex-col gap-6">
      {sections.map(({ id, Component }) => (
        <Component key={id} />
      ))}
    </div>
  );
}

export default AppSettingsScreen;
