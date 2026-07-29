import { useState } from "react";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { useConversationHooks } from "#/hooks/query/use-conversation-hooks";
import { AgentState } from "#/types/agent-state";
import { HooksModalHeader } from "./hooks-modal-header";
import { HooksLoadingState } from "./hooks-loading-state";
import { HooksEmptyState } from "./hooks-empty-state";
import { HookEventItem } from "./hook-event-item";
import { RuntimeWaitingState } from "./runtime-waiting-state";
import { useAgentState } from "#/hooks/use-agent-state";

interface HooksModalProps {
  onClose: () => void;
}

export function HooksModal({ onClose }: HooksModalProps) {
  const { curAgentState } = useAgentState();
  const [expandedEvents, setExpandedEvents] = useState<Record<string, boolean>>(
    {},
  );
  const {
    data: hooks,
    isLoading,
    isError,
    refetch,
    isRefetching,
  } = useConversationHooks();

  const toggleEvent = (eventType: string) => {
    setExpandedEvents((prev) => ({
      ...prev,
      [eventType]: !prev[eventType],
    }));
  };

  const isAgentReady = ![AgentState.LOADING, AgentState.INIT].includes(
    curAgentState,
  );

  return (
    <ModalBackdrop onClose={onClose}>
      <ModalBody
        width="lg"
        className="relative max-h-[80vh] flex flex-col items-start border border-[var(--oh-border)]"
        testID="hooks-modal"
      >
        <HooksModalHeader
          isLoading={isLoading}
          isRefetching={isRefetching}
          onRefresh={refetch}
          onClose={onClose}
        />

        <div className="w-full h-[60vh] overflow-auto rounded-md border border-[var(--oh-border)] bg-surface-raised custom-scrollbar-always">
          {!isAgentReady ? (
            <RuntimeWaitingState testId="hooks-runtime-waiting" />
          ) : isLoading ? (
            <HooksLoadingState />
          ) : isError || !hooks || hooks.length === 0 ? (
            <HooksEmptyState isError={isError} />
          ) : (
            <div className="divide-y divide-[var(--oh-border)]">
              {hooks.map((hookEvent) => {
                const isExpanded =
                  expandedEvents[hookEvent.event_type] || false;

                return (
                  <HookEventItem
                    key={hookEvent.event_type}
                    hookEvent={hookEvent}
                    isExpanded={isExpanded}
                    onToggle={toggleEvent}
                  />
                );
              })}
            </div>
          )}
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
