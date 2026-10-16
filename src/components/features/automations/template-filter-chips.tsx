import type { ReactNode } from "react";
import { Clock, LayoutGrid, Zap, type LucideIcon } from "lucide-react";
import { useTranslation } from "react-i18next";
import { INTEGRATION_CATALOG as MCP_MARKETPLACE } from "@openhands/extensions/integrations";
import { McpLogoBadge } from "#/components/features/mcp-logo-badge";
import { I18nKey } from "#/i18n/declaration";
import type { TemplateTriggerKind } from "#/utils/automation-catalog";
import { dropdownFilterTriggerClassName } from "#/utils/dropdown-classes";
import { getMarketplaceEntryById } from "#/utils/mcp-marketplace-utils";
import { cn } from "#/utils/utils";

/** The Templates page filters: one integration and one trigger kind, combined. */
export interface TemplateFilters {
  integrationId: string | null;
  trigger: TemplateTriggerKind | null;
}

export const EMPTY_TEMPLATE_FILTERS: TemplateFilters = {
  integrationId: null,
  trigger: null,
};

export function hasActiveTemplateFilter(filters: TemplateFilters): boolean {
  return filters.integrationId !== null || filters.trigger !== null;
}

/** The chip and card-badge presentation of each trigger kind. */
export const TRIGGER_KIND_ICON: Record<TemplateTriggerKind, LucideIcon> = {
  cron: Clock,
  event: Zap,
};

export const TRIGGER_KIND_LABEL_KEY: Record<TemplateTriggerKind, I18nKey> = {
  cron: I18nKey.RECOMMENDED_AUTOMATIONS$TRIGGER_CRON,
  event: I18nKey.RECOMMENDED_AUTOMATIONS$TRIGGER_EVENT,
};

const TRIGGER_KINDS: readonly TemplateTriggerKind[] = ["cron", "event"];

function FilterChip({
  active,
  testId,
  onClick,
  children,
}: {
  active: boolean;
  testId: string;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      data-testid={testId}
      aria-pressed={active}
      onClick={onClick}
      className={cn(
        dropdownFilterTriggerClassName,
        "h-8 py-0 text-xs",
        active && "border-contrast/60 bg-contrast/10",
      )}
    >
      {children}
    </button>
  );
}

interface TemplateFilterChipsProps {
  /** The integrations the offered templates are about, in display order. */
  integrationIds: readonly string[];
  filters: TemplateFilters;
  onChange: (next: TemplateFilters) => void;
}

/**
 * Toggle chips above the templates grid: every template, one chip per
 * integration, and one per trigger kind. One integration and one trigger kind
 * can be active at once and combine; clicking an active chip clears it.
 */
export function TemplateFilterChips({
  integrationIds,
  filters,
  onChange,
}: TemplateFilterChipsProps) {
  const { t } = useTranslation("openhands");

  return (
    <div
      role="group"
      aria-label={t(I18nKey.RECOMMENDED_AUTOMATIONS$FILTERS_LABEL)}
      data-testid="template-filters"
      className="flex flex-wrap items-center gap-2"
    >
      <FilterChip
        testId="template-filter-all"
        active={!hasActiveTemplateFilter(filters)}
        onClick={() => onChange(EMPTY_TEMPLATE_FILTERS)}
      >
        <LayoutGrid className="size-3.5 shrink-0" aria-hidden />
        {t(I18nKey.RECOMMENDED_AUTOMATIONS$FILTER_ALL)}
      </FilterChip>
      {integrationIds.map((id) => {
        const entry = getMarketplaceEntryById(id, MCP_MARKETPLACE);
        const active = filters.integrationId === id;
        return (
          <FilterChip
            key={id}
            testId={`template-filter-integration-${id}`}
            active={active}
            onClick={() =>
              onChange({ ...filters, integrationId: active ? null : id })
            }
          >
            <McpLogoBadge entry={entry} size="xs" />
            {entry?.name ?? id}
          </FilterChip>
        );
      })}
      {TRIGGER_KINDS.map((kind) => {
        const Icon = TRIGGER_KIND_ICON[kind];
        const active = filters.trigger === kind;
        return (
          <FilterChip
            key={kind}
            testId={`template-filter-trigger-${kind}`}
            active={active}
            onClick={() =>
              onChange({ ...filters, trigger: active ? null : kind })
            }
          >
            <Icon className="size-3.5 shrink-0" aria-hidden />
            {t(TRIGGER_KIND_LABEL_KEY[kind])}
          </FilterChip>
        );
      })}
    </div>
  );
}
