import { useTranslation } from "react-i18next";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { Typography } from "#/ui/typography";
import { KNOWN_PROFILE_TOOL_DESCRIPTIONS } from "#/constants/profile-tools";

interface ProfileToolListProps {
  catalog: string[];
  selected: string[];
  isDisabled?: boolean;
  onToggle: (name: string, checked: boolean) => void;
}

/**
 * Per-tool toggles for an agent profile's `tools` field.
 *
 * Rows are keyed by the wire tool name — the same identifier the API takes —
 * so a tool this build has no description for is still legible and selectable.
 */
export function ProfileToolList({
  catalog,
  selected,
  isDisabled = false,
  onToggle,
}: ProfileToolListProps) {
  const { t } = useTranslation("openhands");

  return (
    <ul
      data-testid="agent-settings-tool-list"
      className="flex flex-col gap-2.5"
    >
      {catalog.map((name) => {
        const descriptionKey = KNOWN_PROFILE_TOOL_DESCRIPTIONS[name];
        return (
          <li key={name} className="flex flex-col gap-0.5">
            <SettingsSwitch
              testId={`agent-settings-tool-${name}`}
              name={`tool-${name}`}
              isToggled={selected.includes(name)}
              isDisabled={isDisabled}
              onToggle={(checked) => onToggle(name, checked)}
            >
              <span className="font-mono">{name}</span>
            </SettingsSwitch>
            {descriptionKey ? (
              <Typography.Text className="text-xs text-tertiary-alt pl-11">
                {t(descriptionKey)}
              </Typography.Text>
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}
