import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { AutomationCardSkeleton } from "#/components/features/automations/automation-card-skeleton";
import { AutomationGroup } from "#/components/features/automations/automation-group";
import { BackendNotConfigured } from "#/components/features/automations/backend-not-configured";
import { EmptyState } from "#/components/features/automations/empty-state";
import { ErrorState } from "#/components/features/automations/error-state";
import { useAutomations } from "#/hooks/query/use-automations";
import { useAutomationHealth } from "#/hooks/query/use-automation-health";
import { useStartAutomationSetup } from "#/hooks/use-start-automation-setup";
import { I18nKey } from "#/i18n/declaration";

/** Opens the setup page once when the drawer asks to add an automation. */
function LaunchAutomationSetup() {
  const { startSetup } = useStartAutomationSetup();
  const launched = useRef(false);

  useEffect(() => {
    if (launched.current) return;
    launched.current = true;
    startSetup();
  }, [startSetup]);

  return null;
}

interface ConversationOverviewAutomationsPanelProps {
  openAdd: boolean;
}

const NOOP = () => undefined;

/** Reuses the existing Automations list and creation guidance in the drawer. */
export function ConversationOverviewAutomationsPanel({
  openAdd,
}: ConversationOverviewAutomationsPanelProps) {
  const { t } = useTranslation("openhands");
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

  // Every state renders inside the same panel container so the drawer's
  // DOM contract (one `conversation-overview-automations-panel` node) holds
  // regardless of backend health or list contents.
  let body;
  if (isHealthLoading || (health?.status === "ok" && isLoading)) {
    body = <AutomationCardSkeleton />;
  } else if (health?.status !== "ok") {
    body = <BackendNotConfigured onRetry={refetchHealth} />;
  } else if (isError) {
    body = <ErrorState onRetry={refetch} />;
  } else if (!data?.automations.length) {
    body = (
      <>
        <EmptyState />
        {openAdd ? <LaunchAutomationSetup /> : null}
      </>
    );
  } else {
    body = (
      <>
        <AutomationGroup
          title={t(I18nKey.CONVERSATION_PANEL$AUTOMATIONS)}
          count={data.automations.length}
          automations={data.automations}
          view="list"
          onToggle={NOOP}
          onRunNow={NOOP}
          onDelete={NOOP}
          onExport={NOOP}
        />
        {openAdd ? <LaunchAutomationSetup /> : null}
      </>
    );
  }

  return (
    <div data-testid="conversation-overview-automations-panel">{body}</div>
  );
}
