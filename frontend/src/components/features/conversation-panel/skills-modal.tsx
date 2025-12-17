import { useState } from "react";
import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { I18nKey } from "#/i18n/declaration";
import { useConversationSkills } from "#/hooks/query/use-conversation-skills";
import { AgentState } from "#/types/agent-state";
import { Typography } from "#/ui/typography";
import { SkillsModalHeader } from "./skills-modal-header";
import { SkillsLoadingState } from "./skills-loading-state";
import { SkillsEmptyState } from "./skills-empty-state";
import { SkillItem } from "./skill-item";
import { useAgentState } from "#/hooks/use-agent-state";

interface SkillsModalProps {
  onClose: () => void;
}

export function SkillsModal({ onClose }: SkillsModalProps) {
  const { t } = useTranslation();
  const { curAgentState } = useAgentState();
  const [expandedAgents, setExpandedAgents] = useState<Record<string, boolean>>(
    {},
  );
  const {
    data: skills,
    isLoading,
    isError,
    refetch,
    isRefetching,
  } = useConversationSkills();

  const toggleAgent = (agentName: string) => {
    setExpandedAgents((prev) => ({
      ...prev,
      [agentName]: !prev[agentName],
    }));
  };

  const isAgentReady = ![AgentState.LOADING, AgentState.INIT].includes(
    curAgentState,
  );

  return (
    <ModalBackdrop onClose={onClose}>
      <ModalBody
        width="medium"
        className="max-h-[80vh] flex flex-col items-start"
        testID="skills-modal"
      >
        <SkillsModalHeader
          isAgentReady={isAgentReady}
          isLoading={isLoading}
          isRefetching={isRefetching}
          onRefresh={refetch}
        />

        {isAgentReady && (
          <Typography.Text className="text-sm text-gray-400">
            {t(I18nKey.SKILLS_MODAL$WARNING)}
          </Typography.Text>
        )}

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
            (isError || !skills || skills.length === 0) && (
              <SkillsEmptyState isError={isError} />
            )}

          {!isLoading && isAgentReady && skills && skills.length > 0 && (
            <div className="p-2 space-y-3">
              {skills.map((skill) => {
                const isExpanded = expandedAgents[skill.name] || false;

                return (
                  <SkillItem
                    key={skill.name}
                    skill={skill}
                    isExpanded={isExpanded}
                    onToggle={toggleAgent}
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
