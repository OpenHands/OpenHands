import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { cn } from "#/utils/utils";

/** Case-insensitive filter by one or more searchable text fields. */
export function filterByName<T>(
  items: T[],
  query: string,
  getText: (item: T) => Array<string | null | undefined>,
): T[] {
  const needle = query.trim().toLowerCase();
  if (!needle) {
    return items;
  }
  return items.filter((item) =>
    getText(item).some((value) =>
      String(value ?? "")
        .toLowerCase()
        .includes(needle),
    ),
  );
}

export function CloudAiToolbar({
  title,
  onRefresh,
  onCreate,
  onBack,
  createLabel,
  searchValue,
  onSearchChange,
  searchPlaceholder,
}: {
  title: string;
  onRefresh?: () => void;
  onCreate?: () => void;
  onBack?: () => void;
  createLabel?: string;
  searchValue?: string;
  onSearchChange?: (value: string) => void;
  searchPlaceholder?: string;
}) {
  const { t } = useTranslation("openhands");
  return (
    <div className="mb-3 flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        {onBack && (
          <BrandButton
            type="button"
            variant="secondary"
            testId="cloudai-back"
            onClick={onBack}
          >
            {t(I18nKey.CLOUDAI$BACK)}
          </BrandButton>
        )}
        <Typography.Text className="min-w-0 flex-1 truncate text-sm font-semibold text-white">
          {title}
        </Typography.Text>
        {onRefresh && (
          <BrandButton
            type="button"
            variant="secondary"
            testId="cloudai-refresh"
            onClick={onRefresh}
          >
            {t(I18nKey.CLOUDAI$REFRESH)}
          </BrandButton>
        )}
        {onCreate && (
          <BrandButton
            type="button"
            variant="primary"
            testId="cloudai-create"
            onClick={onCreate}
          >
            {createLabel ?? t(I18nKey.CLOUDAI$CREATE)}
          </BrandButton>
        )}
      </div>
      {onSearchChange && (
        <input
          type="search"
          data-testid="cloudai-search"
          value={searchValue ?? ""}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder={
            searchPlaceholder ?? t(I18nKey.CLOUDAI$SEARCH_PLACEHOLDER)
          }
          className={cn(
            "w-full rounded-md border border-[var(--oh-border)]",
            "bg-[var(--oh-surface)] px-3 py-2 text-sm text-white",
            "placeholder:text-[var(--oh-muted)]",
            "focus:outline-none focus:ring-1 focus:ring-[var(--oh-border)]",
          )}
        />
      )}
    </div>
  );
}

export function CloudAiSubNav({
  items,
  activeId,
  onChange,
}: {
  items: { id: string; label: string }[];
  activeId: string;
  onChange: (id: string) => void;
}) {
  return (
    <div
      className="mb-3 flex gap-1 rounded-md border border-[var(--oh-border)] bg-[var(--oh-surface)] p-1"
      data-testid="cloudai-subnav"
    >
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          data-testid={`cloudai-subnav-${item.id}`}
          onClick={() => onChange(item.id)}
          className={cn(
            "flex-1 rounded px-2 py-1.5 text-xs font-medium transition-colors",
            activeId === item.id
              ? "bg-[var(--oh-surface-raised)] text-white"
              : "text-[var(--oh-muted)] hover:text-white",
          )}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

export function CloudAiStatus({
  isLoading,
  isError,
  empty,
  noResults,
}: {
  isLoading?: boolean;
  isError?: boolean;
  empty?: boolean;
  noResults?: boolean;
}) {
  const { t } = useTranslation("openhands");
  if (isLoading) {
    return (
      <div className="flex items-center gap-2 py-6" data-testid="cloudai-loading">
        <LoadingSpinner size="small" />
        <Typography.Text className="text-sm text-[var(--oh-muted)]">
          {t(I18nKey.CLOUDAI$LOADING)}
        </Typography.Text>
      </div>
    );
  }
  if (isError) {
    return (
      <Typography.Text
        className="py-6 text-sm text-red-400"
        testId="cloudai-error"
      >
        {t(I18nKey.CLOUDAI$ERROR)}
      </Typography.Text>
    );
  }
  if (noResults) {
    return (
      <Typography.Text
        className="py-6 text-sm text-[var(--oh-muted)]"
        testId="cloudai-no-results"
      >
        {t(I18nKey.CLOUDAI$NO_RESULTS)}
      </Typography.Text>
    );
  }
  if (empty) {
    return (
      <Typography.Text
        className="py-6 text-sm text-[var(--oh-muted)]"
        testId="cloudai-empty"
      >
        {t(I18nKey.CLOUDAI$NO_ITEMS)}
      </Typography.Text>
    );
  }
  return null;
}

export function CloudAiRow({
  title,
  subtitle,
  badge,
  onOpen,
  onEdit,
  onDelete,
  extraActions,
}: {
  title: string;
  subtitle?: string;
  badge?: string;
  onOpen?: () => void;
  onEdit?: () => void;
  onDelete?: () => void;
  extraActions?: React.ReactNode;
}) {
  const { t } = useTranslation("openhands");
  return (
    <div
      className={cn(
        "group flex items-start gap-2 rounded-lg border border-[var(--oh-border)]",
        "bg-[var(--oh-surface)] px-3 py-2.5 transition-colors",
        onOpen && "hover:border-[var(--oh-muted)] hover:bg-[var(--oh-surface-raised)]",
      )}
    >
      <button
        type="button"
        className="min-w-0 flex-1 text-left"
        onClick={onOpen}
        disabled={!onOpen}
        aria-label={onOpen ? t(I18nKey.CLOUDAI$OPEN) : undefined}
      >
        <div className="flex min-w-0 items-center gap-2">
          <Typography.Text className="block truncate text-sm font-medium text-white">
            {title}
          </Typography.Text>
          {badge && (
            <span className="shrink-0 rounded bg-[var(--oh-surface-raised)] px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-[var(--oh-muted)]">
              {badge}
            </span>
          )}
        </div>
        {subtitle && (
          <Typography.Text className="mt-0.5 block truncate text-xs text-[var(--oh-muted)]">
            {subtitle}
          </Typography.Text>
        )}
      </button>
      <div className="flex shrink-0 flex-wrap items-center justify-end gap-1">
        {extraActions}
        {onEdit && (
          <BrandButton type="button" variant="secondary" onClick={onEdit}>
            {t(I18nKey.CLOUDAI$EDIT)}
          </BrandButton>
        )}
        {onDelete && (
          <BrandButton type="button" variant="danger" onClick={onDelete}>
            {t(I18nKey.CLOUDAI$DELETE)}
          </BrandButton>
        )}
      </div>
    </div>
  );
}

export type CloudAiFormField = {
  key: string;
  label: string;
  type?: string;
  multiline?: boolean;
  checkbox?: boolean;
  required?: boolean;
  help?: string;
  options?: string[];
};

export function CloudAiPromptForm({
  fields,
  values,
  onChange,
  onSubmit,
  onCancel,
  submitLabel,
}: {
  fields: CloudAiFormField[];
  values: Record<string, string>;
  onChange: (key: string, value: string) => void;
  onSubmit: () => void;
  onCancel: () => void;
  submitLabel: string;
}) {
  const { t } = useTranslation("openhands");
  return (
    <form
      className="mb-3 flex flex-col gap-3 rounded-lg border border-[var(--oh-border)] bg-[var(--oh-surface)] p-3"
      data-testid="cloudai-form"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
    >
      {fields.map((field) => {
        if (field.checkbox) {
          return (
            <label
              key={field.key}
              className="flex items-center gap-2 text-sm text-white"
            >
              <input
                type="checkbox"
                className="size-4"
                checked={values[field.key] === "true"}
                onChange={(e) =>
                  onChange(field.key, e.target.checked ? "true" : "false")
                }
              />
              <span>
                {field.label}
                {field.required ? " *" : ""}
              </span>
            </label>
          );
        }

        if (field.options && field.options.length > 0) {
          return (
            <label
              key={field.key}
              className="flex flex-col gap-1 text-xs text-[var(--oh-muted)]"
            >
              <span className="font-medium text-white">
                {field.label}
                {field.required ? " *" : ""}
              </span>
              <select
                className="rounded-md border border-[var(--oh-border)] bg-transparent p-2 text-sm text-white"
                value={values[field.key] ?? ""}
                onChange={(e) => onChange(field.key, e.target.value)}
                required={field.required}
              >
                <option value="">—</option>
                {field.options.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
              {field.help && (
                <span className="text-[11px] text-[var(--oh-muted)]">
                  {field.help}
                </span>
              )}
            </label>
          );
        }

        if (field.multiline) {
          return (
            <label
              key={field.key}
              className="flex flex-col gap-1 text-xs text-[var(--oh-muted)]"
            >
              <span className="font-medium text-white">
                {field.label}
                {field.required ? " *" : ""}
              </span>
              <textarea
                className="min-h-24 rounded-md border border-[var(--oh-border)] bg-transparent p-2 text-sm text-white"
                value={values[field.key] ?? ""}
                onChange={(e) => onChange(field.key, e.target.value)}
                required={field.required}
              />
              {field.help && (
                <span className="text-[11px] text-[var(--oh-muted)]">
                  {field.help}
                </span>
              )}
            </label>
          );
        }

        return (
          <label
            key={field.key}
            className="flex flex-col gap-1 text-xs text-[var(--oh-muted)]"
          >
            <span className="font-medium text-white">
              {field.label}
              {field.required ? " *" : ""}
            </span>
            <input
              type={field.type ?? "text"}
              className="rounded-md border border-[var(--oh-border)] bg-transparent p-2 text-sm text-white"
              value={values[field.key] ?? ""}
              onChange={(e) => onChange(field.key, e.target.value)}
              required={field.required}
            />
            {field.help && (
              <span className="text-[11px] text-[var(--oh-muted)]">
                {field.help}
              </span>
            )}
          </label>
        );
      })}
      <div className="flex gap-2 pt-1">
        <BrandButton type="submit" variant="primary">
          {submitLabel}
        </BrandButton>
        <BrandButton type="button" variant="secondary" onClick={onCancel}>
          {t(I18nKey.BUTTON$CANCEL)}
        </BrandButton>
      </div>
    </form>
  );
}
