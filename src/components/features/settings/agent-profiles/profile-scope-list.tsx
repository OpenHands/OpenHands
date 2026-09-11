import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { Typography } from "#/ui/typography";

export interface ProfileScopeItem {
  name: string;
  description?: string | null;
}

interface ProfileScopeListProps {
  /** Prefix: the list is `<testId>-list`, each row `<testId>-<name>`. */
  testId: string;
  items: ProfileScopeItem[];
  selected: string[];
  isDisabled?: boolean;
  onToggle: (name: string, checked: boolean) => void;
}

/**
 * Per-item toggles for a profile's scope fields (`tools`, `mcp_server_refs`).
 *
 * Rows are keyed by the wire name — the same identifier the API takes — so an
 * entry this build has no description for is still legible and selectable.
 */
export function ProfileScopeList({
  testId,
  items,
  selected,
  isDisabled = false,
  onToggle,
}: ProfileScopeListProps) {
  return (
    <ul data-testid={`${testId}-list`} className="flex flex-col gap-2.5">
      {items.map(({ name, description }) => (
        <li key={name} className="flex flex-col gap-0.5">
          <SettingsSwitch
            testId={`${testId}-${name}`}
            name={`${testId}-${name}`}
            isToggled={selected.includes(name)}
            isDisabled={isDisabled}
            onToggle={(checked) => onToggle(name, checked)}
          >
            <span className="font-mono">{name}</span>
          </SettingsSwitch>
          {description ? (
            <Typography.Text className="text-xs text-tertiary-alt pl-11">
              {description}
            </Typography.Text>
          ) : null}
        </li>
      ))}
    </ul>
  );
}
