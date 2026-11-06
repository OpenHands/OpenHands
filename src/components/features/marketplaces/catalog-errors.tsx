import { useTranslation } from "react-i18next";
import { AlertTriangle } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import type { CatalogError } from "#/types/marketplaces";
import { CATALOG_KIND_LABEL_KEYS } from "./catalog-kinds";

interface CatalogErrorsProps {
  errors: CatalogError[];
}

/** Entries the marketplace lists but Canvas could not read. The rest still loads. */
export function CatalogErrors({ errors }: CatalogErrorsProps) {
  const { t } = useTranslation("openhands");
  if (errors.length === 0) return null;

  return (
    <section
      data-testid="marketplace-catalog-errors"
      className="rounded-xl border border-warning/40 bg-base-secondary p-4"
    >
      <h3 className="flex items-center gap-2 text-sm font-semibold text-warning">
        <AlertTriangle className="h-4 w-4" aria-hidden />
        {t(I18nKey.MARKETPLACES$ERRORS_TITLE, { count: errors.length })}
      </h3>
      <ul className="mt-2 flex flex-col gap-1 text-xs text-tertiary-light">
        {errors.map((error) => (
          <li key={`${error.kind ?? "entry"}:${error.entry}`}>
            <span className="text-contrast">{error.entry}</span>
            {error.kind ? ` (${t(CATALOG_KIND_LABEL_KEYS[error.kind])})` : ""}
            {": "}
            {error.message}
          </li>
        ))}
      </ul>
    </section>
  );
}
