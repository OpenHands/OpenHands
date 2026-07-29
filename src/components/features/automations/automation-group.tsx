import type { Automation } from "#/types/automation";
import { cn } from "#/utils/utils";
import { AutomationCard } from "./automation-card";
import { AutomationListRow } from "./automation-list-row";
import { StatusBadge } from "./status-badge";
import {
  automationListTableClassName,
  type AutomationViewMode,
} from "./automation-view-mode";
import {
  extensionModuleCardGridClassName,
  extensionModuleCardGridContainerClassName,
} from "#/utils/extension-module-card-classes";

interface AutomationGroupProps {
  title: string;
  count: number;
  automations: Automation[];
  view: AutomationViewMode;
  onToggle: (id: string, enabled: boolean) => void;
  onRunNow: (id: string) => void;
  runPendingId?: string | null;
  onDelete: (id: string) => void;
  onExport: (automation: Automation) => void;
  onEdit?: (id: string) => void;
}

export function AutomationGroup({
  title,
  count,
  automations,
  view,
  onToggle,
  onRunNow,
  runPendingId = null,
  onDelete,
  onExport,
  onEdit,
}: AutomationGroupProps) {
  if (automations.length === 0) return null;

  return (
    <section>
      <div className="flex items-center">
        <h2 className="text-base font-semibold text-foreground">{title}</h2>
        <StatusBadge count={count} />
      </div>
      {view === "grid" ? (
        <div className={cn("mt-3", extensionModuleCardGridContainerClassName)}>
          <div className={extensionModuleCardGridClassName}>
            {automations.map((automation) => (
              <AutomationCard
                key={automation.id}
                automation={automation}
                onToggle={onToggle}
                onRunNow={onRunNow}
                isRunPending={runPendingId === automation.id}
                onDelete={onDelete}
                onExport={onExport}
                onEdit={onEdit}
              />
            ))}
          </div>
        </div>
      ) : (
        <div className={cn(automationListTableClassName, "mt-3")}>
          <table className="w-full min-w-full [&>tbody>tr:first-child]:border-t-0">
            <tbody>
              {automations.map((automation) => (
                <AutomationListRow
                  key={automation.id}
                  automation={automation}
                  onToggle={onToggle}
                  onRunNow={onRunNow}
                  isRunPending={runPendingId === automation.id}
                  onDelete={onDelete}
                  onExport={onExport}
                  onEdit={onEdit}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
