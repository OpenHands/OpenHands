import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { I18nKey } from "#/i18n/declaration";
import { useConversationMcps } from "#/hooks/query/use-conversation-mcps";
import { AgentState } from "#/types/agent-state";
import { Typography } from "#/ui/typography";
import { SkillsModalHeader } from "./skills-modal-header";
import { SkillsLoadingState } from "./skills-loading-state";
import { SkillsEmptyState } from "./skills-empty-state";
import { useAgentState } from "#/hooks/use-agent-state";

interface McpsModalProps {
  onClose: () => void;
}

export function McpsModal({ onClose }: McpsModalProps) {
  const { t } = useTranslation();
  const { curAgentState } = useAgentState();
  const {
    data: mcpServers,
    isLoading,
    isError,
    refetch,
    isRefetching,
  } = useConversationMcps();

  const isAgentReady = ![AgentState.LOADING, AgentState.INIT].includes(
    curAgentState,
  );

  return (
    <ModalBackdrop onClose={onClose}>
      <ModalBody
        width="medium"
        className="max-h-[80vh] flex flex-col items-start"
        testID="mcps-modal"
      >
        <SkillsModalHeader
          isAgentReady={isAgentReady}
          isLoading={isLoading}
          isRefetching={isRefetching}
          onRefresh={refetch}
          title={t(I18nKey.MCPS_MODAL$TITLE)}
        />

        <div className="w-full h-[60vh] overflow-auto rounded-md custom-scrollbar-always">
          {!isAgentReady && (
            <div className="w-full h-full flex items-center text-center justify-center text-2xl text-tertiary-light">
              <Typography.Text>
                {t(I18nKey.DIFF_VIEWER$WAITING_FOR_RUNTIME)}
              </Typography.Text>
            </div>
          )}

          {isLoading && <SkillsLoadingState />}

          {!isLoading &&
            isAgentReady &&
            (isError || !mcpServers || mcpServers.length === 0) && (
              <SkillsEmptyState isError={isError} />
            )}

          {!isLoading &&
            isAgentReady &&
            mcpServers &&
            mcpServers.length > 0 && (
              <div className="p-2 space-y-2">
                {mcpServers.map((server) => (
                  <div
                    key={server.name}
                    className="rounded-lg border border-tertiary p-3 space-y-1"
                  >
                    <div className="flex items-center gap-2">
                      <Typography.Text className="font-medium text-sm">
                        {server.name}
                      </Typography.Text>
                      {server.source && (
                        <span className="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primary">
                          {server.source}
                        </span>
                      )}
                      {server.transport && (
                        <span className="text-xs px-2 py-0.5 rounded-full bg-tertiary text-secondary">
                          {server.transport}
                        </span>
                      )}
                    </div>
                    {server.url && (
                      <Typography.Text className="text-xs text-tertiary-light break-all">
                        {server.url}
                      </Typography.Text>
                    )}
                    {server.command && (
                      <Typography.Text className="text-xs text-tertiary-light font-mono">
                        {server.command}
                        {server.args ? ` ${server.args.join(" ")}` : ""}
                      </Typography.Text>
                    )}
                  </div>
                ))}
              </div>
            )}
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
