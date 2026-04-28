import { useState } from "react";
import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { I18nKey } from "#/i18n/declaration";
import { useConversationHooks } from "#/hooks/query/use-conversation-hooks";
import { AgentState } from "#/types/agent-state";
import { Typography } from "#/ui/typography";
import { HooksModalHeader } from "./hooks-modal-header";
import { HooksLoadingState } from "./hooks-loading-state";
import { HooksEmptyState } from "./hooks-empty-state";
import { HookEventItem } from "./hook-event-item";
import { useAgentState } from "#/hooks/use-agent-state";

interface HooksModalProps {
  onClose: () => void;
}

export function HooksModal({ onClose }: HooksModalProps) {
  const { t } = useTranslation();
  const { curAgentState } = useAgentState();
  const [searchQuery, setSearchQuery] = useState("");
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

  const normalizedSearch = searchQuery.trim().toLowerCase();
  const filteredHooks = hooks?.filter((hookEvent) => {
    if (!normalizedSearch) {
      return true;
    }

    return hookEvent.event_type.toLowerCase().includes(normalizedSearch);
  });

  const allExpanded =
    !!filteredHooks?.length &&
    filteredHooks.every((hookEvent) => expandedEvents[hookEvent.event_type]);

  const toggleAllFiltered = () => {
    if (!filteredHooks?.length) {
      return;
    }

    setExpandedEvents((prev) => {
      const next = { ...prev };
      const shouldExpand = !allExpanded;

      filteredHooks.forEach((hookEvent) => {
        next[hookEvent.event_type] = shouldExpand;
      });

      return next;
    });
  };

  return (
    <ModalBackdrop onClose={onClose}>
      <ModalBody
        width="medium"
        className="max-h-[80vh] flex flex-col items-start"
        testID="hooks-modal"
      >
        <HooksModalHeader
          isAgentReady={isAgentReady}
          isLoading={isLoading}
          isRefetching={isRefetching}
          hookCount={filteredHooks?.length ?? 0}
          allExpanded={allExpanded}
          onRefresh={refetch}
          onToggleAll={toggleAllFiltered}
        />

        {isAgentReady && (
          <div className="w-full space-y-2">
            <Typography.Text className="text-sm text-gray-400">
              {t(I18nKey.HOOKS_MODAL$WARNING)}
            </Typography.Text>
            <input
              type="text"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder={t(I18nKey.HOOKS_MODAL$SEARCH_PLACEHOLDER)}
              className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-md text-sm text-gray-100 placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>
        )}

        <div className="w-full h-[60vh] overflow-auto rounded-md custom-scrollbar-always">
          {!isAgentReady && (
            <div className="w-full h-full flex items-center text-center justify-center text-2xl text-tertiary-light">
              <Typography.Text>
                {t(I18nKey.DIFF_VIEWER$WAITING_FOR_RUNTIME)}
              </Typography.Text>
            </div>
          )}

          {isLoading && <HooksLoadingState />}

          {!isLoading &&
            isAgentReady &&
            (isError || !hooks || hooks.length === 0) && (
              <HooksEmptyState isError={isError} />
            )}

          {!isLoading &&
            isAgentReady &&
            filteredHooks &&
            filteredHooks.length > 0 && (
            <div className="p-2 space-y-3">
              {filteredHooks.map((hookEvent) => {
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

          {!isLoading &&
            isAgentReady &&
            hooks &&
            hooks.length > 0 &&
            filteredHooks &&
            filteredHooks.length === 0 && (
              <div className="w-full h-full flex items-center text-center justify-center text-lg text-tertiary-light">
                <Typography.Text>{t(I18nKey.COMMON$NO_RESULTS)}</Typography.Text>
              </div>
            )}
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
