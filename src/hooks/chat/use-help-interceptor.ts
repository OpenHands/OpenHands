import { useCallback } from "react";
import { BUILT_IN_COMMANDS, HELP_COMMAND } from "#/utils/constants";
import { useConversationSkills } from "#/hooks/query/use-conversation-skills";
import { useSlashCommandInfoStore } from "#/stores/slash-command-info-store";

const HELP_PREFIX = `${HELP_COMMAND} `;

/**
 * Derive the slash commands a skill contributes to the autocomplete menu,
 * using the same source the menu itself uses (see use-slash-command.ts):
 * explicit "/" triggers, or a derived "/<name>" for AgentSkills.
 */
export const getSkillDerivedCommands = (
  skills: { name: string; type: string; triggers?: string[] }[] | undefined,
): string[] => {
  if (!skills) return [];
  const builtIn = new Set(BUILT_IN_COMMANDS.map((cmd) => cmd.command));
  const commands = new Set<string>();
  skills.forEach((skill) => {
    const slashTriggers = (skill.triggers ?? []).filter((t) =>
      t.startsWith("/"),
    );
    if (slashTriggers.length > 0) {
      slashTriggers.forEach((trigger) => commands.add(trigger));
    } else if (skill.type === "agentskills") {
      commands.add(`/${skill.name}`);
    }
  });
  return [...commands].filter((command) => !builtIn.has(command)).sort();
};

/**
 * Intercepts "/help" and renders every built-in command with its description
 * plus every skill-derived command in the chat, followed by the note that "/"
 * shows the autocomplete menu. This is the single source of truth for what
 * commands exist. Everything else falls through to `onSubmit`. Passthrough
 * when `conversationId` is null.
 */
export const useHelpInterceptor = (
  conversationId: string | null | undefined,
  onSubmit: (message: string) => void,
) => {
  const { data: skills } = useConversationSkills();
  const addInfo = useSlashCommandInfoStore((s) => s.addInfo);

  return useCallback(
    (message: string) => {
      const trimmed = message.trim();
      const isHelp =
        trimmed === HELP_COMMAND || trimmed.startsWith(HELP_PREFIX);
      if (!conversationId || !isHelp) {
        onSubmit(message);
        return;
      }

      const builtInLines = BUILT_IN_COMMANDS.map(
        (cmd) => `- \`${cmd.command}\` — ${cmd.skill.content ?? ""}`,
      );
      const skillCommands = getSkillDerivedCommands(skills);
      const skillLines =
        skillCommands.length > 0
          ? skillCommands.map((command) => `- \`${command}\``)
          : ["- _No skill commands loaded_"];

      addInfo(
        conversationId,
        HELP_COMMAND,
        [
          "### Built-in commands",
          ...builtInLines,
          "",
          "### Skill commands",
          ...skillLines,
          "",
          "Type `/` to show the autocomplete menu.",
        ].join("\n"),
      );
    },
    [conversationId, onSubmit, skills, addInfo],
  );
};
