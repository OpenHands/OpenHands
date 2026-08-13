import { registerSettingsSection } from "./registry";
import { GeneralSection } from "#/components/features/settings/app-settings/sections/general-section";
import { ConversationTitlesSection } from "#/components/features/settings/app-settings/sections/conversation-titles-section";
import { GitSection } from "#/components/features/settings/app-settings/sections/git-section";
import { AdvancedApplicationSection } from "#/components/features/settings/app-settings/sections/advanced-application-section";

const APP_SETTINGS_PAGE = "/settings/app";

/**
 * Register the built-in Application settings sections.
 *
 * These are ordinary first-party (trusted) React components registered from
 * OSS code — exactly how a future backend-specific or plugin-contributed
 * section would register, only with a `when` predicate. Importing this module
 * for its side effect (see `app-settings.tsx`) performs the registration; it is
 * idempotent by section id, so repeated imports are safe.
 */
export function registerAppSettingsSections(): void {
  registerSettingsSection({
    id: "app.general",
    page: APP_SETTINGS_PAGE,
    order: 10,
    Component: GeneralSection,
  });

  registerSettingsSection({
    id: "app.conversation-titles",
    page: APP_SETTINGS_PAGE,
    order: 20,
    Component: ConversationTitlesSection,
  });

  registerSettingsSection({
    id: "app.git",
    page: APP_SETTINGS_PAGE,
    order: 30,
    Component: GitSection,
  });

  // Backend-specific: only cloud/enterprise backends support these fields. It
  // appears by registration + a `when` predicate instead of an inline
  // `if (isCloud)` block in the page host (issue #16596).
  registerSettingsSection({
    id: "app.advanced",
    page: APP_SETTINGS_PAGE,
    order: 40,
    when: (context) => context.backendKind === "cloud",
    Component: AdvancedApplicationSection,
  });
}

registerAppSettingsSections();
