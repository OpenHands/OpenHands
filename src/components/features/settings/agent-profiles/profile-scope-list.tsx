import { Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { I18nKey } from "#/i18n/declaration";
import { settingsListIconActionButtonClassName } from "#/utils/settings-list-classes";
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
  onToggle?: (name: string, checked: boolean) => void;
  onDelete?: (name: string) => void;
  deletableNames?: string[];
  isDeleteDisabled?: boolean;
}

/**
 * Per-item toggles for a profile scope field such as `mcp_server_refs`.
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
  onDelete,
  deletableNames,
  isDeleteDisabled = false,
}: ProfileScopeListProps) {
  const { t } = useTranslation("openhands");
  return (
    <ul data-testid={`${testId}-list`} className="flex flex-col gap-2.5">
      {items.map(({ name, description }) => (
        <li key={name} className="flex flex-col gap-0.5">
          <div className="flex items-center justify-between gap-3">
            <SettingsSwitch
              testId={`${testId}-${name}`}
              name={`${testId}-${name}`}
              isToggled={selected.includes(name)}
              isDisabled={isDisabled || !onToggle}
              onToggle={(checked) => onToggle?.(name, checked)}
            >
              <span className="font-mono">{name}</span>
            </SettingsSwitch>
            {onDelete && deletableNames?.includes(name) && (
              <button
                type="button"
                data-testid={`${testId}-delete-${name}`}
                aria-label={`${t(I18nKey.BUTTON$DELETE)} ${name}`}
                onClick={() => onDelete(name)}
                disabled={isDeleteDisabled}
                className={settingsListIconActionButtonClassName}
              >
                <Trash2 aria-hidden="true" className="size-4" />
              </button>
            )}
          </div>
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
