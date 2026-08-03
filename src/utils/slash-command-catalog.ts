import type { SlashCommandItem } from "#/types/slash-command";
import { BUILT_IN_COMMANDS } from "#/utils/constants";

interface BuildSlashCommandCatalogOptions {
  skills?: SlashCommandItem["skill"][];
  isSkillsLoading?: boolean;
  isCloud: boolean;
  hasConversation: boolean;
  agentKind?: "openhands" | "acp" | null;
  supportsManualCondensation?: boolean;
}

const isAvailable = (
  availability: (typeof BUILT_IN_COMMANDS)[number]["availability"],
  isCloud: boolean,
  hasConversation: boolean,
  agentKind: "openhands" | "acp" | null | undefined,
  supportsManualCondensation: boolean | undefined,
) => {
  if (availability === "always") return true;
  if (availability === "conversation") return hasConversation;
  if (availability === "confirmation") {
    return hasConversation && (isCloud || agentKind === "openhands");
  }
  if (availability === "manual-condensation-conversation") {
    // Manual condensation is a local Agent Server capability. Derive it from
    // the running agent's serialized condenser; unknown capabilities and
    // Cloud stay undiscoverable so help never promises an unusable action.
    return !isCloud && hasConversation && supportsManualCondensation === true;
  }
  if (availability === "local-conversation") {
    return !isCloud && hasConversation;
  }
  if (availability === "local-openhands-conversation") {
    return !isCloud && hasConversation && agentKind === "openhands";
  }
  return false;
};

export const getSkillSlashCommandItems = (
  skills: SlashCommandItem["skill"][] | undefined,
): SlashCommandItem[] => {
  if (!skills) return [];

  return skills.flatMap((skill) => {
    const slashTriggers = (skill.triggers ?? []).filter((trigger) =>
      trigger.startsWith("/"),
    );
    if (slashTriggers.length > 0) {
      return slashTriggers.map((command) => ({ skill, command }));
    }
    return skill.type === "agentskills"
      ? [{ skill, command: `/${skill.name}` }]
      : [];
  });
};

export const buildSlashCommandCatalog = ({
  skills,
  isSkillsLoading = false,
  isCloud,
  hasConversation,
  agentKind,
  supportsManualCondensation,
}: BuildSlashCommandCatalogOptions): SlashCommandItem[] => {
  const builtIns = BUILT_IN_COMMANDS.filter((item) =>
    isAvailable(
      item.availability,
      isCloud,
      hasConversation,
      agentKind,
      supportsManualCondensation,
    ),
  );
  if (isSkillsLoading) return builtIns;

  // Interceptors reserve every frontend-owned command even when it is not
  // advertised in the current context. Letting a skill claim an unavailable
  // built-in would advertise a command that the frontend later swallows.
  const seen = new Set(BUILT_IN_COMMANDS.map((item) => item.command));
  const skillItems = getSkillSlashCommandItems(skills)
    .filter((item) => {
      if (seen.has(item.command)) return false;
      seen.add(item.command);
      return true;
    })
    .sort((left, right) => left.command.localeCompare(right.command));

  return [...builtIns, ...skillItems];
};
