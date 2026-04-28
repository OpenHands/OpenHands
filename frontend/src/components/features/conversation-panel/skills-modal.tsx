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
  const [searchQuery, setSearchQuery] = useState("");
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

  const normalizedSearch = searchQuery.trim().toLowerCase();
  const filteredSkills = skills?.filter((skill) => {
    if (!normalizedSearch) {
      return true;
    }

    return (
      skill.name.toLowerCase().includes(normalizedSearch) ||
      skill.type.toLowerCase().includes(normalizedSearch)
    );
  });

  const allExpanded =
    !!filteredSkills?.length &&
    filteredSkills.every((skill) => expandedAgents[skill.name]);

  const toggleAllFiltered = () => {
    if (!filteredSkills?.length) {
      return;
    }

    setExpandedAgents((prev) => {
      const next = { ...prev };
      const shouldExpand = !allExpanded;

      filteredSkills.forEach((skill) => {
        next[skill.name] = shouldExpand;
      });

      return next;
    });
  };

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
          skillCount={filteredSkills?.length ?? 0}
          allExpanded={allExpanded}
          onRefresh={refetch}
          onToggleAll={toggleAllFiltered}
        />

        {isAgentReady && (
          <div className="w-full space-y-2">
            <Typography.Text className="text-sm text-gray-400">
              {t(I18nKey.SKILLS_MODAL$WARNING)}
            </Typography.Text>
            <input
              type="text"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder={t(I18nKey.SKILLS_MODAL$SEARCH_PLACEHOLDER)}
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

          {isLoading && <SkillsLoadingState />}

          {!isLoading &&
            isAgentReady &&
            (isError || !skills || skills.length === 0) && (
              <SkillsEmptyState isError={isError} />
            )}

          {!isLoading && isAgentReady && filteredSkills && filteredSkills.length > 0 && (
            <div className="p-2 space-y-3">
              {filteredSkills.map((skill) => {
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

          {!isLoading &&
            isAgentReady &&
            skills &&
            skills.length > 0 &&
            filteredSkills &&
            filteredSkills.length === 0 && (
              <div className="w-full h-full flex items-center text-center justify-center text-lg text-tertiary-light">
                <Typography.Text>{t(I18nKey.COMMON$NO_RESULTS)}</Typography.Text>
              </div>
            )}
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
