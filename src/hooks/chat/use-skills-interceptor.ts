import { useCallback } from "react";
import { SKILLS_COMMAND } from "#/utils/constants";
import { useConversationSkills } from "#/hooks/query/use-conversation-skills";
import { useSlashCommandInfoStore } from "#/stores/slash-command-info-store";

const SKILLS_PREFIX = `${SKILLS_COMMAND} `;

/**
 * Intercepts "/skills" and renders the list already fetched by
 * `useConversationSkills()` inline in the chat — skills, hooks, and MCPs.
 * Display-only: no backend call beyond the existing fetch. Everything else
 * falls through to `onSubmit`. Passthrough when `conversationId` is null.
 */
export const useSkillsInterceptor = (
  conversationId: string | null | undefined,
  onSubmit: (message: string) => void,
) => {
  const { data: skills } = useConversationSkills();
  const addInfo = useSlashCommandInfoStore((s) => s.addInfo);

  return useCallback(
    (message: string) => {
      const trimmed = message.trim();
      const isSkills =
        trimmed === SKILLS_COMMAND || trimmed.startsWith(SKILLS_PREFIX);
      if (!conversationId || !isSkills) {
        onSubmit(message);
        return;
      }

      const lines =
        skills && skills.length > 0
          ? skills.map((skill) => {
              const detail = skill.description ?? skill.content ?? "";
              return `- \`${skill.name}\` (${skill.type})${detail ? ` — ${detail}` : ""}`;
            })
          : ["- _No skills loaded_"];

      addInfo(
        conversationId,
        SKILLS_COMMAND,
        ["### Loaded skills", ...lines].join("\n"),
      );
    },
    [conversationId, onSubmit, skills, addInfo],
  );
};
