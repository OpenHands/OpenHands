import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { I18nKey } from "#/i18n/declaration";
import { getAgentServerWorkingDir } from "#/api/agent-server-config";
import { useConversationSkills } from "#/hooks/query/use-conversation-skills";
import {
  groupSkillsByScope,
  SKILL_SCOPE_ORDER,
  type SkillScope,
} from "#/utils/skill-scope";
import { SkillsModalHeader } from "./skills-modal-header";
import { SkillsModalSection } from "./skills-modal-section";
import { SkillsLoadingState } from "./skills-loading-state";
import { SkillsEmptyState } from "./skills-empty-state";
import { SkillItem } from "./skill-item";

interface SkillsModalProps {
  onClose: () => void;
}

const SECTION_TITLE_KEY: Record<SkillScope, I18nKey> = {
  project: I18nKey.SKILLS_MODAL$SECTION_PROJECT,
  personal: I18nKey.SKILLS_MODAL$SECTION_USER,
  public: I18nKey.SKILLS_MODAL$SECTION_PUBLIC,
};

export function SkillsModal({ onClose }: SkillsModalProps) {
  const { t } = useTranslation("openhands");
  const projectDir = getAgentServerWorkingDir();
  const [expandedAgents, setExpandedAgents] = useState<Record<string, boolean>>(
    {},
  );
  // Scope the catalog to this conversation's attached workspace so the listed
  // skills match the project skills actually loaded into the conversation.
  const {
    data: skills,
    isLoading,
    isError,
    refetch,
    isRefetching,
  } = useConversationSkills();

  const groupedSkills = useMemo(
    () => (skills ? groupSkillsByScope(skills, projectDir) : null),
    [skills, projectDir],
  );

  const toggleAgent = (agentName: string) => {
    setExpandedAgents((prev) => ({
      ...prev,
      [agentName]: !prev[agentName],
    }));
  };

  return (
    <ModalBackdrop onClose={onClose}>
      <ModalBody
        width="lg"
        className="relative max-h-[80vh] flex flex-col items-start border border-[var(--oh-border)]"
        testID="skills-modal"
      >
        <SkillsModalHeader
          isLoading={isLoading}
          isRefetching={isRefetching}
          onRefresh={refetch}
          onClose={onClose}
        />

        <div className="w-full h-[60vh] overflow-auto rounded-md border border-[var(--oh-border)] bg-surface-raised custom-scrollbar-always">
          {isLoading ? (
            <SkillsLoadingState />
          ) : isError || !skills || skills.length === 0 ? (
            <SkillsEmptyState isError={isError} />
          ) : (
            groupedSkills && (
              <div className="divide-y divide-[var(--oh-border)]">
                {SKILL_SCOPE_ORDER.map((scope) => {
                  const scopedSkills = groupedSkills[scope];
                  if (scopedSkills.length === 0) {
                    return null;
                  }

                  return (
                    <SkillsModalSection
                      key={scope}
                      title={t(SECTION_TITLE_KEY[scope])}
                      count={scopedSkills.length}
                    >
                      {scopedSkills.map((skill) => {
                        const isExpanded = expandedAgents[skill.name] || false;

                        return (
                          <SkillItem
                            key={`${scope}-${skill.name}`}
                            skill={skill}
                            isExpanded={isExpanded}
                            onToggle={toggleAgent}
                          />
                        );
                      })}
                    </SkillsModalSection>
                  );
                })}
              </div>
            )
          )}
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
