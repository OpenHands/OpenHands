import { type ReactNode, useState } from "react";
import { ChevronDown, ListFilter } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { EnumFilterDropdown } from "#/components/shared/filters/enum-filter-dropdown";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import type { DashboardSpec } from "#/manifests/automation-interface";
import {
  DASHBOARD_TAG_FILTER_ID,
  type DashboardCreatedByValue,
  type DashboardSortValue,
  type DashboardStatusValue,
  type DashboardTriggerValue,
} from "#/manifests/types";
import { dropdownFilterTriggerClassName } from "#/utils/dropdown-classes";
import { cn } from "#/utils/utils";

function FilterField({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs font-medium text-muted">{label}</span>
      {children}
    </div>
  );
}

function toLabelMap<T extends string>(
  options: readonly { value: T; label: string }[],
): Record<T, string> {
  return Object.fromEntries(
    options.map((option) => [option.value, option.label]),
  ) as Record<T, string>;
}

interface AutomationsDashboardControlsProps {
  spec: DashboardSpec;
  status: DashboardStatusValue;
  trigger: DashboardTriggerValue;
  createdBy: DashboardCreatedByValue;
  canFilterByCreator: boolean;
  /** The selected tag, or null for any. */
  tag: string | null;
  /** The org's tags; the tag filter is hidden while this is empty. */
  tagOptions: readonly string[];
  sort: DashboardSortValue;
  onStatusChange: (value: DashboardStatusValue) => void;
  onTriggerChange: (value: DashboardTriggerValue) => void;
  onCreatedByChange: (value: DashboardCreatedByValue) => void;
  onTagChange: (value: string | null) => void;
  onSortChange: (value: DashboardSortValue) => void;
}

// No tag is empty, so the neutral option can stand for "any tag" in the
// dropdown's string values.
const ANY_TAG_VALUE = "";

/**
 * One Filters trigger that nests the manifest-declared status, trigger,
 * created-by, tag, and sort dropdowns. Which filters exist, their options, and
 * every caption stay the manifest's (the tag filter's options are the org's
 * tags); the predicates and comparators behind the values stay the host's.
 */
export function AutomationsDashboardControls({
  spec,
  status,
  trigger,
  createdBy,
  canFilterByCreator,
  tag,
  tagOptions,
  sort,
  onStatusChange,
  onTriggerChange,
  onCreatedByChange,
  onTagChange,
  onSortChange,
}: AutomationsDashboardControlsProps) {
  const { t } = useTranslation("openhands");
  const [open, setOpen] = useState(false);
  const containerRef = useClickOutsideElement<HTMLDivElement>(() =>
    setOpen(false),
  );

  const statusFilter = spec.filters.find((filter) => filter.id === "status");
  const triggerFilter = spec.filters.find((filter) => filter.id === "trigger");
  const createdByFilter = canFilterByCreator
    ? spec.filters.find((filter) => filter.id === "created_by")
    : undefined;
  const tagFilter =
    tagOptions.length > 0
      ? spec.filters.find((filter) => filter.id === DASHBOARD_TAG_FILTER_ID)
      : undefined;
  // A selected tag stays listed after its last automation drops it, so the
  // selection is always visible and can be cleared.
  const tagValues = [
    ANY_TAG_VALUE,
    ...tagOptions,
    ...(tag !== null && !tagOptions.includes(tag) ? [tag] : []),
  ];
  const activeCount = [
    statusFilter && status !== statusFilter.options[0]?.value,
    triggerFilter && trigger !== triggerFilter.options[0]?.value,
    createdByFilter && createdBy !== createdByFilter.options[0]?.value,
    tagFilter && tag !== null,
    sort !== spec.sort.default,
  ].filter(Boolean).length;
  const filtersLabel = t(I18nKey.AUTOMATIONS$FILTERS);
  const defaultStatus = statusFilter?.options[0]?.value;
  const defaultTrigger = triggerFilter?.options[0]?.value;
  const defaultCreatedBy = createdByFilter?.options[0]?.value;

  const resetAll = () => {
    if (defaultStatus) onStatusChange(defaultStatus);
    if (defaultTrigger) onTriggerChange(defaultTrigger);
    if (defaultCreatedBy) onCreatedByChange(defaultCreatedBy);
    if (tagFilter) onTagChange(null);
    onSortChange(spec.sort.default);
  };

  return (
    <div
      ref={containerRef}
      className="relative shrink-0 w-auto"
      data-testid="automations-filters"
    >
      <button
        type="button"
        data-testid="dropdown-trigger"
        aria-haspopup="true"
        aria-expanded={open}
        aria-label={filtersLabel}
        onClick={() => setOpen((prev) => !prev)}
        className={cn(
          dropdownFilterTriggerClassName,
          "h-9 py-0",
          activeCount > 0 && "border-contrast/60 bg-contrast/10",
        )}
      >
        <ListFilter className="h-4 w-4 shrink-0" aria-hidden />
        <span className="whitespace-nowrap">{filtersLabel}</span>
        {activeCount > 0 ? (
          <span className="rounded-full bg-contrast px-1.5 text-[11px] font-medium text-contrast-foreground">
            {activeCount}
          </span>
        ) : null}
        <ChevronDown
          className={cn(
            "h-4 w-4 shrink-0 text-tertiary-alt transition-transform",
            open && "rotate-180",
          )}
          aria-hidden
        />
      </button>

      {open ? (
        <div
          role="group"
          data-testid="automations-filters-menu"
          aria-label={filtersLabel}
          className="absolute right-0 top-full z-50 mt-1 flex min-w-[16rem] w-max flex-col gap-3 overflow-visible rounded-md bg-tertiary p-3 context-menu-box-shadow"
        >
          {statusFilter ? (
            <FilterField label={statusFilter.label}>
              <EnumFilterDropdown
                testId="automations-filter-status"
                value={status}
                onChange={onStatusChange}
                options={statusFilter.options.map((option) => option.value)}
                labelByValue={toLabelMap(statusFilter.options)}
                ariaLabel={statusFilter.label}
                fullWidth
              />
            </FilterField>
          ) : null}
          {triggerFilter ? (
            <FilterField label={triggerFilter.label}>
              <EnumFilterDropdown
                testId="automations-filter-trigger"
                value={trigger}
                onChange={onTriggerChange}
                options={triggerFilter.options.map((option) => option.value)}
                labelByValue={toLabelMap(triggerFilter.options)}
                ariaLabel={triggerFilter.label}
                fullWidth
              />
            </FilterField>
          ) : null}
          {createdByFilter ? (
            <FilterField label={createdByFilter.label}>
              <EnumFilterDropdown
                testId="automations-filter-created-by"
                value={createdBy}
                onChange={onCreatedByChange}
                options={createdByFilter.options.map((option) => option.value)}
                labelByValue={toLabelMap(createdByFilter.options)}
                ariaLabel={createdByFilter.label}
                fullWidth
              />
            </FilterField>
          ) : null}
          {tagFilter ? (
            <FilterField label={tagFilter.label}>
              <EnumFilterDropdown
                testId="automations-filter-tag"
                value={tag ?? ANY_TAG_VALUE}
                onChange={(value) =>
                  onTagChange(value === ANY_TAG_VALUE ? null : value)
                }
                options={tagValues}
                // Tags label themselves; only the neutral option has a caption.
                labelByValue={{ [ANY_TAG_VALUE]: tagFilter.allLabel }}
                ariaLabel={tagFilter.label}
                fullWidth
              />
            </FilterField>
          ) : null}
          <FilterField label={spec.sort.label}>
            <EnumFilterDropdown
              testId="automations-sort"
              value={sort}
              onChange={onSortChange}
              options={spec.sort.options.map((option) => option.value)}
              labelByValue={toLabelMap(spec.sort.options)}
              ariaLabel={spec.sort.label}
              fullWidth
            />
          </FilterField>
          {activeCount > 0 ? (
            <button
              type="button"
              data-testid="automations-filters-reset"
              onClick={resetAll}
              className="self-start text-sm text-muted hover:text-foreground"
            >
              {t(I18nKey.AUTOMATIONS$RESET_ALL)}
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
