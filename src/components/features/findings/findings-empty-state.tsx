/**
 * Empty / forbidden / error states for the findings page.
 * @spec PROJETOSIN-188 — findings-empty-state
 */

import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";
import { cn } from "#/utils/utils";

export function FindingsEmptyNoEngagement() {
  const { t } = useTranslation("openhands");
  return (
    <div
      data-testid="findings-empty-no-engagement"
      role="status"
      className={extensionModuleEmptyStateClassName}
    >
      <p className="text-sm text-white">
        {t(I18nKey.FINDINGS$EMPTY_NO_ENGAGEMENT)}
      </p>
      <p className="mt-1 text-xs text-tertiary-light">
        {t(I18nKey.FINDINGS$EMPTY_NO_ENGAGEMENT_HINT)}
      </p>
    </div>
  );
}

export function FindingsEmpty() {
  const { t } = useTranslation("openhands");
  return (
    <div
      data-testid="findings-empty"
      role="status"
      className={extensionModuleEmptyStateClassName}
    >
      <p className="text-sm text-white">{t(I18nKey.FINDINGS$EMPTY)}</p>
      <p className="mt-1 text-xs text-tertiary-light">
        {t(I18nKey.FINDINGS$EMPTY_HINT)}
      </p>
    </div>
  );
}

export function FindingsFilteredEmpty({ onClear }: { onClear: () => void }) {
  const { t } = useTranslation("openhands");
  return (
    <div
      data-testid="findings-filtered-empty"
      role="status"
      className={cn(extensionModuleEmptyStateClassName, "border-dashed")}
    >
      <p className="text-sm text-white">
        {t(I18nKey.FINDINGS$NO_FILTER_MATCHES)}
      </p>
      <button
        type="button"
        data-testid="findings-clear-filters"
        className="mt-3 text-sm text-[var(--oh-color-primary)] underline-offset-2 hover:underline"
        onClick={onClear}
      >
        {t(I18nKey.FINDINGS$CLEAR_FILTERS)}
      </button>
    </div>
  );
}

export function FindingsForbidden() {
  const { t } = useTranslation("openhands");
  return (
    <div
      data-testid="findings-forbidden"
      role="status"
      className={extensionModuleEmptyStateClassName}
    >
      <p className="text-sm text-white">{t(I18nKey.FINDINGS$FORBIDDEN)}</p>
      <p className="mt-1 text-xs text-tertiary-light">
        {t(I18nKey.FINDINGS$FORBIDDEN_HINT)}
      </p>
    </div>
  );
}

export function FindingsError({ onRetry }: { onRetry: () => void }) {
  const { t } = useTranslation("openhands");
  return (
    <div
      data-testid="findings-error"
      role="alert"
      className={extensionModuleEmptyStateClassName}
    >
      <p className="text-sm text-white">{t(I18nKey.FINDINGS$ERROR)}</p>
      <div className="mt-4 flex justify-center">
        <BrandButton type="button" variant="secondary" onClick={onRetry}>
          {t(I18nKey.FINDINGS$ERROR_RETRY)}
        </BrandButton>
      </div>
    </div>
  );
}

export function FindingsLoading() {
  const { t } = useTranslation("openhands");
  return (
    <div
      data-testid="findings-loading"
      aria-busy="true"
      aria-label={t(I18nKey.FINDINGS$LOADING)}
      className="flex flex-col gap-3"
    >
      <div className="flex flex-wrap gap-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <div
            key={i}
            className="h-8 w-24 animate-pulse rounded-md bg-[var(--oh-surface-raised)]"
          />
        ))}
      </div>
      {Array.from({ length: 6 }).map((_, i) => (
        <div
          key={i}
          className="h-12 w-full animate-pulse rounded-md bg-[var(--oh-surface-raised)]"
        />
      ))}
    </div>
  );
}
