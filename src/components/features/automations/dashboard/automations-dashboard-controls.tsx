import { type ReactNode, useState } from "react";
import { ChevronDown, ListFilter } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { EnumFilterDropdown } from "#/components/shared/filters/enum-filter-dropdown";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import type { DashboardSpec } from "#/manifests/automation-interface";
import type {
  DashboardSortValue,
  DashboardStatusValue,
  DashboardTriggerValue,
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
  sort: DashboardSortValue;
  onStatusChange: (value: DashboardStatusValue) => void;
  onTriggerChange: (value: DashboardTriggerValue) => void;
  onSortChange: (value: DashboardSortValue) => void;
}

/**
 * One Filters trigger that nests the manifest-declared status, trigger, and
 * sort dropdowns. Which filters exist, their options, and every caption stay
 * the manifest's; the predicates and comparators behind the values stay the
 * host's.
 */
export function AutomationsDashboardControls({
  spec,
  status,
  trigger,
  sort,
  onStatusChange,
  onTriggerChange,
  onSortChange,
}: AutomationsDashboardControlsProps) {
  const { t } = useTranslation("openhands");
  const [open, setOpen] = useState(false);
  const containerRef = useClickOutsideElement<HTMLDivElement>(() =>
    setOpen(false),
  );

  const statusFilter = spec.filters.find((filter) => filter.id === "status");
  const triggerFilter = spec.filters.find((filter) => filter.id === "trigger");
  const activeCount = [
    statusFilter && status !== statusFilter.options[0]?.value,
    triggerFilter && trigger !== triggerFilter.options[0]?.value,
    sort !== spec.sort.default,
  ].filter(Boolean).length;
  const filtersLabel = t(I18nKey.AUTOMATIONS$FILTERS);
  const defaultStatus = statusFilter?.options[0]?.value;
  const defaultTrigger = triggerFilter?.options[0]?.value;

  const resetAll = () => {
    if (defaultStatus) onStatusChange(defaultStatus);
    if (defaultTrigger) onTriggerChange(defaultTrigger);
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
          activeCount > 0 && "border-white/60 bg-white/10",
        )}
      >
        <ListFilter className="h-4 w-4 shrink-0" aria-hidden />
        <span className="whitespace-nowrap">{filtersLabel}</span>
        {activeCount > 0 ? (
          <span className="rounded-full bg-white px-1.5 text-[11px] font-medium text-black">
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
          className="absolute right-0 top-full z-50 mt-1 flex min-w-[16rem] w-max flex-col gap-3 overflow-visible rounded-[6px] bg-tertiary p-3 context-menu-box-shadow"
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
