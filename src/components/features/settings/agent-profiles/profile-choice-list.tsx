import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

export interface ProfileChoice {
  name: string;
  label?: string;
  description?: string | null;
  selectable?: boolean;
}

interface ProfileChoiceListProps {
  label: string;
  name: string;
  testId: string;
  choices: ProfileChoice[];
  selected: string;
  onSelect: (name: string) => void;
  onEdit?: (name: string) => void;
  onDelete?: (name: string) => void;
  disabled?: boolean;
}

export function ProfileChoiceList({
  label,
  name,
  testId,
  choices,
  selected,
  onSelect,
  onEdit,
  onDelete,
  disabled = false,
}: ProfileChoiceListProps) {
  const { t } = useTranslation("openhands");

  return (
    <fieldset data-testid={testId} className="min-w-0">
      <legend className="mb-2 text-sm">{label}</legend>
      <div className="max-h-56 overflow-y-auto rounded-lg border border-border">
        {choices.map((choice) => (
          <div
            key={choice.name}
            className="flex min-w-0 items-center gap-2 border-b border-border px-3 py-2 last:border-b-0"
          >
            <label className="flex min-w-0 flex-1 cursor-pointer items-center gap-3">
              <input
                type="radio"
                name={name}
                value={choice.name}
                checked={selected === choice.name}
                disabled={disabled}
                onChange={() => onSelect(choice.name)}
                data-testid={`${testId}-option-${choice.name}`}
                className="shrink-0 accent-primary"
              />
              <span className="flex min-w-0 flex-col">
                <span
                  className="truncate text-sm text-foreground"
                  title={choice.label ?? choice.name}
                >
                  {choice.label ?? choice.name}
                </span>
                {choice.description ? (
                  <span
                    className="truncate text-xs text-muted"
                    title={choice.description}
                  >
                    {choice.description}
                  </span>
                ) : null}
              </span>
            </label>
            {choice.selectable !== false && onEdit ? (
              <button
                type="button"
                onClick={() => onEdit(choice.name)}
                disabled={disabled}
                className="shrink-0 rounded px-2 py-1 text-xs text-foreground hover:bg-interactive-hover disabled:opacity-50"
                data-testid={`${testId}-edit-${choice.name}`}
                aria-label={`${t(I18nKey.BUTTON$EDIT)} ${choice.name}`}
              >
                {t(I18nKey.BUTTON$EDIT)}
              </button>
            ) : null}
            {choice.selectable !== false && onDelete ? (
              <button
                type="button"
                onClick={() => onDelete(choice.name)}
                disabled={disabled}
                className="shrink-0 rounded px-2 py-1 text-xs text-feedback-error hover:bg-interactive-hover disabled:opacity-50"
                data-testid={`${testId}-delete-${choice.name}`}
                aria-label={`${t(I18nKey.BUTTON$DELETE)} ${choice.name}`}
              >
                {t(I18nKey.BUTTON$DELETE)}
              </button>
            ) : null}
          </div>
        ))}
      </div>
    </fieldset>
  );
}
