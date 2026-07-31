import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { AutomationCardSkeleton } from "#/components/features/automations/automation-card-skeleton";
import { AutomationGroup } from "#/components/features/automations/automation-group";
import { AddAutomationModal } from "#/components/features/automations/add-automation-modal";
import { BackendNotConfigured } from "#/components/features/automations/backend-not-configured";
import { EmptyState } from "#/components/features/automations/empty-state";
import { ErrorState } from "#/components/features/automations/error-state";
import { useAutomations } from "#/hooks/query/use-automations";
import { useAutomationHealth } from "#/hooks/query/use-automation-health";
import { I18nKey } from "#/i18n/declaration";

interface ConversationOverviewAutomationsPanelProps {
  openAdd: boolean;
}

const NOOP = () => undefined;

/** Reuses the existing Automations list and creation guidance in the drawer. */
export function ConversationOverviewAutomationsPanel({
  openAdd,
}: ConversationOverviewAutomationsPanelProps) {
  const { t } = useTranslation("openhands");
  const [isAddModalOpen, setIsAddModalOpen] = useState(openAdd);
  const {
    data: health,
    isLoading: isHealthLoading,
    refetch: refetchHealth,
  } = useAutomationHealth();
  const { data, isLoading, isError, refetch } = useAutomations({
    limit: 50,
    offset: 0,
    enabled: health?.status === "ok",
  });

  useEffect(() => {
    if (openAdd) setIsAddModalOpen(true);
  }, [openAdd]);

  if (isHealthLoading || (health?.status === "ok" && isLoading)) {
    return (
      <div data-testid="conversation-overview-automations-panel">
        <AutomationCardSkeleton />
      </div>
    );
  }
  if (health?.status !== "ok") {
    return <BackendNotConfigured onRetry={refetchHealth} />;
  }
  if (isError) {
    return <ErrorState onRetry={refetch} />;
  }
  if (!data?.automations.length) {
    return (
      <>
        <EmptyState />
        <AddAutomationModal
          isOpen={isAddModalOpen}
          onClose={() => setIsAddModalOpen(false)}
        />
      </>
    );
  }

  return (
    <div data-testid="conversation-overview-automations-panel">
      <AutomationGroup
        title={t(I18nKey.AUTOMATIONS$TITLE)}
        count={data.automations.length}
        automations={data.automations}
        view="list"
        onToggle={NOOP}
        onRunNow={NOOP}
        onDelete={NOOP}
        onExport={NOOP}
      />
      <AddAutomationModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
      />
    </div>
  );
}
