import React from "react";
import { useTranslation } from "react-i18next";
import { createPortal } from "react-dom";
import { OpenHandsAction } from "#/types/core/actions";
import { OpenHandsObservation } from "#/types/core/observations";
import {
  isOpenHandsAction,
  isOpenHandsObservation,
  isOpenHandsEvent,
  isAgentStateChangeObservation,
  isFinishAction,
} from "#/types/core/guards";
import { EventMessage } from "./event-message";
import { ChatMessage } from "./chat-message";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { LaunchMicroagentModal } from "./microagent/launch-microagent-modal";
import { useUserConversation } from "#/hooks/query/use-user-conversation";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useCreateConversationAndSubscribeMultiple } from "#/hooks/use-create-conversation-and-subscribe-multiple";
import {
  MicroagentStatus,
  EventMicroagentStatus,
} from "#/types/microagent-status";
import { AgentState } from "#/types/agent-state";
import { getFirstPRUrl } from "#/utils/parse-pr-url";
import MemoryIcon from "#/icons/memory_icon.svg?react";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";

const isErrorEvent = (evt: unknown): evt is { error: true; message: string } =>
  typeof evt === "object" &&
  evt !== null &&
  "error" in evt &&
  evt.error === true;

const isAgentStatusError = (evt: unknown): boolean =>
  isOpenHandsEvent(evt) &&
  isAgentStateChangeObservation(evt) &&
  evt.extras.agent_state === AgentState.ERROR;

interface MessagesProps {
  messages: (OpenHandsAction | OpenHandsObservation)[];
  isAwaitingUserConfirmation: boolean;
}

export const Messages: React.FC<MessagesProps> = React.memo(
  ({ messages, isAwaitingUserConfirmation }) => {
    const {
      createConversationAndSubscribe,
      isPending,
      unsubscribeFromConversation,
    } = useCreateConversationAndSubscribeMultiple();
    const { getOptimisticUserMessage } = useOptimisticUserMessageStore();
    const { conversationId } = useConversationId();
    const { data: conversation } = useUserConversation(conversationId);
    const { data: activeConversation } = useActiveConversation();

    // TODO: Hide microagent actions for V1 conversations
    // This is a temporary measure and may be re-enabled in the future
    const isV1Conversation = activeConversation?.conversation_version === "V1";

    const optimisticUserMessage = getOptimisticUserMessage();

    const [selectedEventId, setSelectedEventId] = React.useState<number | null>(
      null,
    );
    const [showLaunchMicroagentModal, setShowLaunchMicroagentModal] =
      React.useState(false);
    const [microagentStatuses, setMicroagentStatuses] = React.useState<
      EventMicroagentStatus[]
    >([]);

    const { t } = useTranslation();

    const actionHasObservationPair = React.useCallback(
      (event: OpenHandsAction | OpenHandsObservation): boolean => {
        if (isOpenHandsAction(event)) {
          return !!messages.some(
            (msg) => isOpenHandsObservation(msg) && msg.cause === event.id,
          );
        }

        return false;
      },
      [messages],
    );

    const getMicroagentStatusForEvent = React.useCallback(
      (eventId: number): MicroagentStatus | null => {
        const statusEntry = microagentStatuses.find(
          (entry) => entry.eventId === eventId,
        );
        return statusEntry?.status || null;
      },
      [microagentStatuses],
    );

    const getMicroagentConversationIdForEvent = React.useCallback(
      (eventId: number): string | undefined => {
        const statusEntry = microagentStatuses.find(
          (entry) => entry.eventId === eventId,
        );
        return statusEntry?.conversationId || undefined;
      },
      [microagentStatuses],
    );

    const getMicroagentPRUrlForEvent = React.useCallback(
      (eventId: number): string | undefined => {
        const statusEntry = microagentStatuses.find(
          (entry) => entry.eventId === eventId,
        );
        return statusEntry?.prUrl || undefined;
      },
      [microagentStatuses],
    );

    const handleMicroagentEvent = React.useCallback(
      (socketEvent: unknown, microagentConversationId: string) => {
        if (isErrorEvent(socketEvent) || isAgentStatusError(socketEvent)) {
          setMicroagentStatuses((prev) =>
            prev.map((statusEntry) =>
              statusEntry.conversationId === microagentConversationId
                ? { ...statusEntry, status: MicroagentStatus.ERROR }
                : statusEntry,
            ),
          );
        } else if (
          isOpenHandsEvent(socketEvent) &&
          isAgentStateChangeObservation(socketEvent)
        ) {
          // Handle completion states
          if (
            socketEvent.extras.agent_state === AgentState.FINISHED ||
            socketEvent.extras.agent_state === AgentState.AWAITING_USER_INPUT
          ) {
            setMicroagentStatuses((prev) =>
              prev.map((statusEntry) =>
                statusEntry.conversationId === microagentConversationId
                  ? { ...statusEntry, status: MicroagentStatus.COMPLETED }
                  : statusEntry,
              ),
            );

            unsubscribeFromConversation(microagentConversationId);
          }
        } else if (
          isOpenHandsEvent(socketEvent) &&
          isFinishAction(socketEvent)
        ) {
          // Check if the finish action contains a PR URL
          const prUrl = getFirstPRUrl(socketEvent.args.final_thought || "");
          if (prUrl) {
            setMicroagentStatuses((prev) =>
              prev.map((statusEntry) =>
                statusEntry.conversationId === microagentConversationId
                  ? {
                      ...statusEntry,
                      status: MicroagentStatus.COMPLETED,
                      prUrl,
                    }
                  : statusEntry,
              ),
            );
          }

          unsubscribeFromConversation(microagentConversationId);
        } else {
          // For any other event, transition from WAITING to CREATING if still waiting
          setMicroagentStatuses((prev) => {
            const currentStatus = prev.find(
              (entry) => entry.conversationId === microagentConversationId,
            )?.status;

            if (currentStatus === MicroagentStatus.WAITING) {
              return prev.map((statusEntry) =>
                statusEntry.conversationId === microagentConversationId
                  ? { ...statusEntry, status: MicroagentStatus.CREATING }
                  : statusEntry,
              );
            }
            return prev; // No change needed
          });
        }
      },
      [setMicroagentStatuses, unsubscribeFromConversation],
    );

    const handleLaunchMicroagent = (
      query: string,
      target: string,
      triggers: string[],
    ) => {
      const conversationInstructions = `Target file: ${target}\n\nDescription: ${query}\n\nTriggers: ${triggers.join(", ")}`;
      if (
        !conversation ||
        !conversation.selected_repository ||
        !conversation.selected_branch ||
        !conversation.git_provider ||
        !selectedEventId
      ) {
        return;
      }

      createConversationAndSubscribe({
        query,
        conversationInstructions,
        repository: {
          name: conversation.selected_repository,
          branch: conversation.selected_branch,
          gitProvider: conversation.git_provider,
        },
        onSuccessCallback: (newConversationId: string) => {
          setShowLaunchMicroagentModal(false);
          // Update status with conversation ID - start with WAITING
          setMicroagentStatuses((prev) => [
            ...prev.filter((status) => status.eventId !== selectedEventId),
            {
              eventId: selectedEventId,
              conversationId: newConversationId,
              status: MicroagentStatus.WAITING,
            },
          ]);
        },
        onEventCallback: (socketEvent: unknown, newConversationId: string) => {
          handleMicroagentEvent(socketEvent, newConversationId);
        },
      });
    };

    return (
      <>
        {messages.map((message, index) => (
          <EventMessage
            key={index}
            event={message}
            hasObservationPair={actionHasObservationPair(message)}
            isAwaitingUserConfirmation={isAwaitingUserConfirmation}
            isLastMessage={messages.length - 1 === index}
            microagentStatus={getMicroagentStatusForEvent(message.id)}
            microagentConversationId={getMicroagentConversationIdForEvent(
              message.id,
            )}
            microagentPRUrl={getMicroagentPRUrlForEvent(message.id)}
            actions={
              conversation?.selected_repository && !isV1Conversation
                ? [
                    {
                      icon: (
                        <MemoryIcon className="w-[14px] h-[14px] text-white" />
                      ),
                      onClick: () => {
                        setSelectedEventId(message.id);
                        setShowLaunchMicroagentModal(true);
                      },
                      tooltip: t("MICROAGENT$ADD_TO_MEMORY"),
                    },
                  ]
                : undefined
            }
            isInLast10Actions={messages.length - 1 - index < 10}
          />
        ))}

        {optimisticUserMessage && (
          <ChatMessage type="user" message={optimisticUserMessage} />
        )}
        {conversation?.selected_repository &&
          !isV1Conversation &&
          showLaunchMicroagentModal &&
          selectedEventId &&
          createPortal(
            <LaunchMicroagentModal
              onClose={() => setShowLaunchMicroagentModal(false)}
              onLaunch={handleLaunchMicroagent}
              selectedRepo={
                conversation.selected_repository.split("/").pop() || ""
              }
              eventId={selectedEventId}
              isLoading={isPending}
            />,
            document.getElementById("modal-portal-exit") || document.body,
          )}
      </>
    );
  },
  (prevProps, nextProps) => {
    // Prevent re-renders if messages are the same length
    if (prevProps.messages.length !== nextProps.messages.length) {
      return false;
    }

    return true;
  },
);

Messages.displayName = "Messages";
