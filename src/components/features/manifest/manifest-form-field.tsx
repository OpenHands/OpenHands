import { useTranslation } from "react-i18next";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { GitRepoDropdown } from "#/components/features/home/git-repo-dropdown";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { I18nKey } from "#/i18n/declaration";
import { formControlMultilineFieldClassName } from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";
import type { GitRepository } from "#/types/git";
import type {
  SetupFieldOption,
  SetupFormField as SetupFormFieldDefinition,
} from "#/manifests/types";

export interface SetupFormFieldProps {
  /** The record key the field is declared under, and what `{{form.x}}` reads. */
  name: string;
  field: SetupFormFieldDefinition;
  value: string;
  /** Already-resolved copy: local checks and service errors look the same here. */
  error?: string;
  /** Declared options, or the ones the deployment supplied. */
  options: SetupFieldOption[];
  /** The picked repository, kept so the picker can show what is selected. */
  repository: GitRepository | null;
  disabled: boolean;
  onChange: (value: string) => void;
  onRepositoryChange: (repository: GitRepository | null) => void;
  onBlur: () => void;
}

/**
 * Render one manifest-declared field.
 *
 * The host knows how to render a field *type*; what any field means is the
 * manifest's business. Every field's own copy comes from the manifest and is
 * never translated by the host; the only host string is the format hint the
 * repository fallback below needs, because no manifest declares it.
 */
export function SetupFormField({
  name,
  field,
  value,
  error,
  options,
  repository,
  disabled,
  onChange,
  onRepositoryChange,
  onBlur,
}: SetupFormFieldProps) {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const testId = `setup-field-${name}`;
  const help = <p className="text-xs text-[var(--oh-muted)]">{field.help}</p>;

  // Listing a user's repositories is a cloud-backend capability: `GitService`
  // answers with an empty page on any other backend, so the picker would offer
  // a source that can never respond. Recommended automations are themselves
  // local-only, which makes that the common case rather than the edge one.
  const canListRepositories = backend.kind === "cloud";

  if (field.type === "repo-picker" && canListRepositories) {
    return (
      <div className="flex w-full flex-col gap-2.5">
        <FieldLabel field={field} />
        <GitRepoDropdown
          provider={field.provider ?? "github"}
          value={repository?.id ?? null}
          repositoryName={repository?.full_name ?? value ?? null}
          placeholder={field.placeholder}
          disabled={disabled}
          onChange={(selected) => {
            onRepositoryChange(selected ?? null);
            onChange(selected?.full_name ?? "");
            onBlur();
          }}
        />
        <FieldError testId={testId} error={error} />
        {help}
      </div>
    );
  }

  // A timezone field declares no options of its own: the accepted zones are the
  // deployment's, so it renders as a list once they are known and as a plain
  // input when they are not.
  if (
    field.type === "select" ||
    (field.type === "timezone" && options.length > 0)
  ) {
    return (
      <div className="flex w-full flex-col gap-2.5">
        <SettingsDropdownInput
          testId={testId}
          name={name}
          label={<FieldLabelText field={field} />}
          items={options.map((option) => ({
            key: option.value,
            label: option.label,
          }))}
          selectedKey={value || undefined}
          placeholder={field.placeholder}
          isDisabled={disabled}
          required={field.required}
          onSelectionChange={(key) => {
            onChange(key === null ? "" : String(key));
            onBlur();
          }}
        />
        <FieldError testId={testId} error={error} />
        {help}
      </div>
    );
  }

  if (field.type === "textarea") {
    return (
      <label className="flex w-full flex-col gap-2.5">
        <FieldLabelText field={field} />
        <textarea
          data-testid={testId}
          name={name}
          rows={4}
          value={value}
          placeholder={field.placeholder}
          disabled={disabled}
          aria-invalid={!!error}
          onChange={(event) => onChange(event.target.value)}
          onBlur={onBlur}
          className={cn(
            formControlMultilineFieldClassName,
            error && "border-red-500",
          )}
        />
        <FieldError testId={testId} error={error} />
        {help}
      </label>
    );
  }

  // `text`, `cron`, an unresolved `timezone` and a repository that cannot be
  // browsed are all single-line strings to the host; only the manifest and the
  // service know what a cron expression or a repository name means. A manifest
  // states the format of everything it declares except the repository, whose
  // shape the host derives, so that one hint is host copy.
  const placeholder =
    field.placeholder ??
    (field.type === "repo-picker"
      ? t(I18nKey.SETUP$REPOSITORY_PLACEHOLDER)
      : undefined);

  return (
    <div className="flex w-full flex-col gap-2.5">
      <SettingsInput
        testId={testId}
        name={name}
        type="text"
        label={field.label}
        value={value}
        placeholder={placeholder}
        isDisabled={disabled}
        showRequiredTag={field.required}
        error={error}
        onChange={onChange}
        onBlur={onBlur}
      />
      {help}
    </div>
  );
}

function FieldLabelText({ field }: { field: SetupFormFieldDefinition }) {
  return (
    <span className="flex items-center gap-2 text-sm">
      {field.label}
      {field.required && (
        <span className="text-sm leading-none text-red-400" aria-hidden>
          *
        </span>
      )}
    </span>
  );
}

function FieldLabel({ field }: { field: SetupFormFieldDefinition }) {
  return (
    <div className="flex items-center gap-2">
      <FieldLabelText field={field} />
    </div>
  );
}

function FieldError({ testId, error }: { testId: string; error?: string }) {
  if (!error) return null;
  return (
    <p
      role="alert"
      data-testid={`${testId}-error`}
      className="-mt-1 text-xs text-red-400"
    >
      {error}
    </p>
  );
}
