import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { getConversationTools } from "#/api/conversation-configuration-service";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useConversationMcpCapabilities } from "#/hooks/query/use-conversation-configuration-capability";
import { CONVERSATION_CONFIGURATION_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import {
  MODAL_MAX_WIDTH_VIEWPORT,
  ModalBody,
} from "#/components/shared/modals/modal-body";
import { SystemMessageHeader } from "./system-message-modal/system-message-header";
import { TabNavigation } from "./system-message-modal/tab-navigation";
import { TabContent } from "./system-message-modal/tab-content";
import { SystemMessageForModal } from "#/utils/system-message-adapter";
import { cn } from "#/utils/utils";

interface SystemMessageModalProps {
  isOpen: boolean;
  onClose: () => void;
  systemMessage: SystemMessageForModal | null;
  conversationId?: string;
}

function ModalContent({
  isOpen,
  onClose,
  systemMessage,
  liveTools,
  toolsPending,
  toolsError,
}: Omit<SystemMessageModalProps, "systemMessage" | "conversationId"> & {
  systemMessage: SystemMessageForModal;
  liveTools?: Record<string, unknown>[];
  toolsPending?: boolean;
  toolsError?: boolean;
}) {
  const { t } = useTranslation("openhands");
  const [activeTab, setActiveTab] = useState<"system" | "tools">("system");
  const [expandedTools, setExpandedTools] = useState<Record<number, boolean>>(
    {},
  );
  const displayedMessage = liveTools
    ? { ...systemMessage, tools: liveTools }
    : systemMessage;

  const toggleTool = (index: number) => {
    setExpandedTools((prev) => ({ ...prev, [index]: !prev[index] }));
  };

  return (
    isOpen && (
      <ModalBackdrop onClose={onClose}>
        <ModalBody
          width="lg"
          className={cn(
            "relative dialog-content-height flex flex-col items-start border border-border",
            MODAL_MAX_WIDTH_VIEWPORT,
          )}
          testID="system-message-modal"
        >
          <SystemMessageHeader
            agentClass={systemMessage.agent_class}
            openhandsVersion={systemMessage.openhands_version}
            onClose={onClose}
          />
          <div className="w-full">
            <TabNavigation
              activeTab={activeTab}
              onTabChange={setActiveTab}
              hasTools={
                toolsPending ||
                toolsError ||
                liveTools !== undefined ||
                !!displayedMessage.tools?.length
              }
            />
            <div className="conversation-dialog-reading-height overflow-auto rounded-md border border-border bg-surface-raised custom-scrollbar-always">
              {activeTab === "tools" && toolsPending ? (
                <p role="status" className="p-4 text-muted">
                  {t(I18nKey.HOME$LOADING)}
                </p>
              ) : activeTab === "tools" && toolsError ? (
                <p role="alert" className="p-4 text-feedback-error">
                  {t(I18nKey.SYSTEM_MESSAGE_MODAL$TOOLS_LOAD_FAILED)}
                </p>
              ) : (
                <TabContent
                  activeTab={activeTab}
                  systemMessage={displayedMessage}
                  expandedTools={expandedTools}
                  onToggleTool={toggleTool}
                />
              )}
            </div>
          </div>
        </ModalBody>
      </ModalBackdrop>
    )
  );
}

function LiveSystemMessageModal({
  conversationId,
  ...props
}: SystemMessageModalProps & {
  conversationId: string;
  systemMessage: SystemMessageForModal;
}) {
  const { backend } = useActiveBackend();
  const { canListTools, isLoading: capabilitiesLoading } =
    useConversationMcpCapabilities(props.isOpen);
  const { data, isFetching, isError } = useQuery({
    queryKey: CONVERSATION_CONFIGURATION_QUERY_KEYS.tools(
      backend.id,
      backend.connectionRevision,
      conversationId,
    ),
    queryFn: () => getConversationTools(conversationId),
    enabled: props.isOpen && canListTools,
    staleTime: 0,
    retry: false,
    meta: { disableToast: true },
  });
  return (
    <ModalContent
      {...props}
      liveTools={
        canListTools ? data?.tools.map((tool) => ({ ...tool })) : undefined
      }
      toolsPending={capabilitiesLoading || (canListTools && isFetching)}
      toolsError={canListTools && isError}
    />
  );
}

export function SystemMessageModal(props: SystemMessageModalProps) {
  if (!props.systemMessage) return null;
  if (props.conversationId) {
    return (
      <LiveSystemMessageModal
        {...props}
        systemMessage={props.systemMessage}
        conversationId={props.conversationId}
      />
    );
  }
  return <ModalContent {...props} systemMessage={props.systemMessage} />;
}
