import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { SettingsDropdownInput } from "../settings-dropdown-input";
import {
  AVAILABLE_COLOR_THEMES,
  type ColorThemeKey,
  applyColorTheme,
  persistColorTheme,
  readPersistedColorTheme,
} from "#/themes/color-themes";

export function ThemeInput() {
  const { t } = useTranslation("openhands");

  const handleSelectionChange = React.useCallback((key: React.Key | null) => {
    if (!key) return;
    const next = key as ColorThemeKey;
    applyColorTheme(next);
    persistColorTheme(next);
  }, []);

  return (
    <SettingsDropdownInput
      testId="color-theme-input"
      name="color-theme-input"
      label={t(I18nKey.SETTINGS$COLOR_THEME)}
      items={AVAILABLE_COLOR_THEMES.map((theme) => ({
        key: theme.key,
        label: theme.label,
      }))}
      defaultSelectedKey={readPersistedColorTheme()}
      onSelectionChange={handleSelectionChange}
      isClearable={false}
      wrapperClassName="w-full min-w-0"
    />
  );
}
