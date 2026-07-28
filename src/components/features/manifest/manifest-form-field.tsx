import { SettingsInput } from "#/components/features/settings/settings-input";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { GitRepoDropdown } from "#/components/features/home/git-repo-dropdown";
import { formControlMultilineFieldClassName } from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";
import type { GitRepository } from "#/types/git";
import type {
  ManifestFieldOption,
  ManifestFormField as ManifestFormFieldDefinition,
} from "#/manifests/types";

export interface ManifestFormFieldProps {
  field: ManifestFormFieldDefinition;
  value: string;
  /** Already-resolved copy: local checks and service errors look the same here. */
  error?: string;
  /** Declared options, or the ones the deployment supplied. */
  options: ManifestFieldOption[];
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
 * manifest's business. Every user-visible string here comes from the manifest,
 * which is why none of them are translated by the host.
 */
export function ManifestFormField({
  field,
  value,
  error,
  options,
  repository,
  disabled,
  onChange,
  onRepositoryChange,
  onBlur,
}: ManifestFormFieldProps) {
  const testId = `manifest-field-${field.name}`;
  const help = <p className="text-xs text-[var(--oh-muted)]">{field.help}</p>;

  if (field.type === "repo-picker") {
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

  if (field.type === "select") {
    return (
      <div className="flex w-full flex-col gap-2.5">
        <SettingsDropdownInput
          testId={testId}
          name={field.name}
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
          name={field.name}
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

  // `text` and `cron` are both single-line strings to the host; only the
  // manifest and the service know what a cron expression means.
  return (
    <div className="flex w-full flex-col gap-2.5">
      <SettingsInput
        testId={testId}
        name={field.name}
        type="text"
        label={field.label}
        value={value}
        placeholder={field.placeholder}
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

function FieldLabelText({ field }: { field: ManifestFormFieldDefinition }) {
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

function FieldLabel({ field }: { field: ManifestFormFieldDefinition }) {
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
