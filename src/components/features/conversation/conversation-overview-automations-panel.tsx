import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Repeat } from "lucide-react";
import { AutomationCardSkeleton } from "#/components/features/automations/automation-card-skeleton";
import { AutomationGroup } from "#/components/features/automations/automation-group";
import { AddAutomationModal } from "#/components/features/automations/add-automation-modal";
import { BackendNotConfigured } from "#/components/features/automations/backend-not-configured";
import { EmptyState } from "#/components/features/automations/empty-state";
import { ErrorState } from "#/components/features/automations/error-state";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useAutomations } from "#/hooks/query/use-automations";
import { useAutomationHealth } from "#/hooks/query/use-automation-health";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useLaunchSkillInChat } from "#/hooks/use-launch-skill-in-chat";
import { I18nKey } from "#/i18n/declaration";
import { useConversationOverviewDrawerOptional } from "./conversation-overview-drawer-context";

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
        <TurnIntoAutomationAction />
        <EmptyState />
        <AddAutomationModal
          isOpen={isAddModalOpen}
          onClose={() => setIsAddModalOpen(false)}
        />
      </>
    );
  } else {
    body = (
      <>
        <TurnIntoAutomationAction />
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
        <AddAutomationModal
          isOpen={isAddModalOpen}
          onClose={() => setIsAddModalOpen(false)}
        />
      </>
    );
  }

  return (
    <div data-testid="conversation-overview-automations-panel">{body}</div>
  );
}

/**
 * Entry point for turning THIS conversation into an automation.
 *
 * Reuses the existing chat-based creation flow (the same one behind
 * "Create automation"): clicking seeds a composer with a prompt derived from
 * this conversation's title, where the agent drafts goal/trigger/etc. The
 * user reviews and edits the draft before anything is created — nothing is
 * sent, created or enabled by clicking here.
 */
function TurnIntoAutomationAction() {
  const { t } = useTranslation("openhands");
  const launchInChat = useLaunchSkillInChat();
  const overviewDrawer = useConversationOverviewDrawerOptional();
  const { data: conversation } = useActiveConversation();

  const handleTurnIntoAutomation = () => {
    launchInChat(
      t(I18nKey.AUTOMATIONS$CREATE_FROM_CONVERSATION_PROMPT, {
        title:
          conversation?.title?.trim() ||
          t(I18nKey.AUTOMATIONS$CREATE_FROM_CONVERSATION_UNTITLED),
      }),
      () => overviewDrawer?.closeDrawer(),
    );
  };

  return (
    <section
      data-testid="turn-into-automation-action"
      className="w-full rounded-lg border border-[var(--oh-border)] bg-[var(--oh-surface)] p-4"
    >
      <p className="text-xs leading-relaxed text-muted">
        {t(I18nKey.AUTOMATIONS$TURN_INTO_AUTOMATION_DESC)}
      </p>
      <BrandButton
        type="button"
        variant="secondary"
        testId="turn-conversation-into-automation-button"
        className="mt-3 whitespace-nowrap"
        onClick={handleTurnIntoAutomation}
        startContent={<Repeat className="size-4" aria-hidden />}
      >
        {t(I18nKey.AUTOMATIONS$TURN_INTO_AUTOMATION)}
      </BrandButton>
    </section>
  );
}
