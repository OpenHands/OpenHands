import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useAgentProfiles } from "#/hooks/query/use-agent-profiles";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";

interface Props {
  value: string | null;
  onChange: (value: string | null) => void;
}

/** One selector for automation setup and editing, backed by the profile library. */
export function AutomationAgentProfileSelector({ value, onChange }: Props) {
  const { t } = useTranslation("openhands");
  const { data, isLoading, isError } = useAgentProfiles();
  const profiles = data?.profiles ?? [];
  const normalizedValue = value || null;
  const selected = normalizedValue
    ? profiles.find((profile) => profile.id === normalizedValue)
    : undefined;
  const defaultLabel = t(I18nKey.SETTINGS$PROFILE_DEFAULT);
  const label =
    selected?.name ??
    (normalizedValue
      ? t(I18nKey.AUTOMATIONS$UNAVAILABLE_AGENT_PROFILE, {
          id: normalizedValue,
        })
      : defaultLabel);
  const items = [
    { key: "__default__", label: defaultLabel },
    ...profiles.flatMap((profile) =>
      profile.id ? [{ key: profile.id, label: profile.name }] : [],
    ),
    ...(normalizedValue && !selected ? [{ key: normalizedValue, label }] : []),
  ];
  return (
    <SettingsDropdownInput
      testId="automation-agent-profile"
      name="agent_profile_id"
      label={t(I18nKey.CHAT$AGENT_PROFILE_PLACEHOLDER)}
      items={items}
      selectedKey={normalizedValue ?? "__default__"}
      isClearable
      isLoading={isLoading}
      isDisabled={isError}
      onSelectionChange={(key) =>
        onChange(key && key !== "__default__" ? String(key) : null)
      }
    />
  );
}
