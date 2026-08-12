import { EnumFilterDropdown } from "#/components/shared/filters/enum-filter-dropdown";
import type { DashboardSpec } from "#/manifests/automation-interface";
import type {
  DashboardSortValue,
  DashboardStatusValue,
  DashboardTriggerValue,
} from "#/manifests/types";

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
 * The manifest-declared filter and sort dropdowns, in the manifest's order.
 * Which filters exist, their options, and every caption are the manifest's;
 * the predicates and comparators behind the values are the host's.
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
  return (
    <>
      {spec.filters.map((filter) =>
        filter.id === "status" ? (
          <EnumFilterDropdown
            key={filter.id}
            testId="automations-filter-status"
            value={status}
            onChange={onStatusChange}
            options={filter.options.map((option) => option.value)}
            labelByValue={toLabelMap(filter.options)}
            ariaLabel={filter.label}
          />
        ) : (
          <EnumFilterDropdown
            key={filter.id}
            testId="automations-filter-trigger"
            value={trigger}
            onChange={onTriggerChange}
            options={filter.options.map((option) => option.value)}
            labelByValue={toLabelMap(filter.options)}
            ariaLabel={filter.label}
          />
        ),
      )}
      <EnumFilterDropdown
        testId="automations-sort"
        value={sort}
        onChange={onSortChange}
        options={spec.sort.options.map((option) => option.value)}
        labelByValue={toLabelMap(spec.sort.options)}
        ariaLabel={spec.sort.label}
      />
    </>
  );
}
