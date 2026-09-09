import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { StatusBadge } from "#/components/features/automations/status-badge";
import { LoopDefinitionCard } from "#/components/features/loops/loop-definition-card";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useLoopDefinitions, useLoopTriggers } from "#/hooks/query/use-loops";
import { I18nKey } from "#/i18n/declaration";
import {
  extensionModuleCardGridClassName,
  extensionModuleCardGridContainerClassName,
} from "#/utils/extension-module-card-classes";
import { cn } from "#/utils/utils";

interface LoopsDashboardSectionProps {
  searchQuery: string;
}

function LoopsDashboardSectionBody({
  searchQuery,
}: LoopsDashboardSectionProps) {
  const { t } = useTranslation("openhands");
  const definitionsQuery = useLoopDefinitions();
  const triggersQuery = useLoopTriggers();
  const query = searchQuery.trim().toLowerCase();
  const definitions = useMemo(() => {
    const items = definitionsQuery.data ?? [];
    if (!query) return items;
    return items.filter((definition) =>
      definition.name.toLowerCase().includes(query),
    );
  }, [definitionsQuery.data, query]);
  const triggersByDefinition = useMemo(() => {
    const map = new Map<string, string>();
    for (const trigger of triggersQuery.data ?? []) {
      if (!trigger.enabled || map.has(trigger.loop_definition_id)) continue;
      map.set(trigger.loop_definition_id, trigger.id);
    }
    return map;
  }, [triggersQuery.data]);

  if (definitions.length === 0) return null;

  return (
    <section data-testid="loops-dashboard-section">
      <div className="flex items-center">
        <h2 className="text-base font-semibold text-foreground">
          {t(I18nKey.LOOPS$NAV)}
        </h2>
        <StatusBadge count={definitions.length} />
      </div>
      <div className={cn("mt-3", extensionModuleCardGridContainerClassName)}>
        <div className={extensionModuleCardGridClassName}>
          {definitions.map((definition) => (
            <LoopDefinitionCard
              key={definition.id}
              definition={definition}
              runTriggerId={triggersByDefinition.get(definition.id) ?? null}
            />
          ))}
        </div>
      </div>
    </section>
  );
}

/** Local-only loop definitions on the Automate dashboard. */
export function LoopsDashboardSection({
  searchQuery,
}: LoopsDashboardSectionProps) {
  const active = useActiveBackend();
  if (active.backend.kind !== "local") return null;
  return <LoopsDashboardSectionBody searchQuery={searchQuery} />;
}
