import { useState } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { GitRepoDropdown } from "#/components/features/home/git-repo-dropdown";
import { I18nKey } from "#/i18n/declaration";
import type { SetupFormField as SetupFormFieldDefinition } from "#/manifests/types";

export interface SetupRepositoryListProps {
  name: string;
  field: SetupFormFieldDefinition;
  values: string[];
  /** Whether this backend can list the user's repositories to pick from. */
  canListRepositories: boolean;
  placeholder?: string;
  disabled: boolean;
  onChange: (values: string[]) => void;
  onBlur: () => void;
}

/**
 * A repository field that collects several repositories.
 *
 * One automation polling several repositories is the shape the entry asked
 * for; the alternative is one automation each, with the trigger label, tone and
 * schedule restated every time. Added repositories are listed above the input
 * that adds them, each removable, because the list is the answer and the input
 * is only how it is built.
 *
 * A repository already in the list is not added twice: the entry would poll it
 * twice per run for one result.
 */
export function SetupRepositoryList({
  name,
  field,
  values,
  canListRepositories,
  placeholder,
  disabled,
  onChange,
  onBlur,
}: SetupRepositoryListProps) {
  const { t } = useTranslation("openhands");
  const [draft, setDraft] = useState("");

  const add = (value: string) => {
    const entry = value.trim();
    if (!entry || values.includes(entry)) return;
    onChange([...values, entry]);
    setDraft("");
    onBlur();
  };

  const remove = (value: string) => {
    onChange(values.filter((item) => item !== value));
    onBlur();
  };

  return (
    <div
      className="flex w-full flex-col gap-2"
      data-testid={`setup-list-${name}`}
    >
      {values.length > 0 && (
        <ul className="flex flex-col gap-1.5">
          {values.map((value) => (
            <li
              key={value}
              className="flex items-center justify-between gap-2 rounded-md border border-[var(--oh-border)] px-3 py-1.5"
            >
              <span className="truncate text-sm">{value}</span>
              <BrandButton
                testId={`setup-list-${name}-remove-${value}`}
                type="button"
                variant="ghost-danger"
                isDisabled={disabled}
                ariaLabel={t(I18nKey.COMMON$REMOVE)}
                onClick={() => remove(value)}
              >
                {t(I18nKey.COMMON$REMOVE)}
              </BrandButton>
            </li>
          ))}
        </ul>
      )}

      {canListRepositories ? (
        <GitRepoDropdown
          provider={field.provider ?? "github"}
          value={null}
          repositoryName={null}
          placeholder={placeholder}
          disabled={disabled}
          onChange={(selected) => {
            if (selected?.full_name) add(selected.full_name);
          }}
        />
      ) : (
        <div className="flex items-end gap-2">
          <div className="grow">
            <SettingsInput
              testId={`setup-field-${name}`}
              name={name}
              // The field's own label is rendered above the list, so this input
              // shows none of its own; it is the control that adds an entry.
              // It still answers to that label, or it would be an input a
              // screen reader announces as nothing at all.
              label=""
              ariaLabel={field.label}
              type="text"
              value={draft}
              placeholder={placeholder}
              isDisabled={disabled}
              onChange={setDraft}
              onKeyDown={(event) => {
                if (event.key !== "Enter") return;
                // Enter adds a repository rather than submitting the dialog,
                // which would create the automation from a half-built list.
                event.preventDefault();
                add(draft);
              }}
              // A repository typed but not added is still an answer the user
              // gave: leaving the input commits it, rather than dropping it on
              // the way to a Continue that reads the list alone.
              onBlur={() => add(draft)}
            />
          </div>
          <BrandButton
            testId={`setup-list-${name}-add`}
            type="button"
            variant="secondary"
            isDisabled={disabled || draft.trim() === ""}
            onClick={() => add(draft)}
          >
            {t(I18nKey.BUTTON$ADD)}
          </BrandButton>
        </div>
      )}
    </div>
  );
}
