import type { TFunction } from "i18next";
import type {
  BuiltInSlashCommandItem,
  SlashCommandItem,
} from "#/types/slash-command";
import { getSkillDescription, stripMarkdown } from "#/utils/skill-description";

/** Resolve frontend-owned built-ins and backend/catalog skill data uniformly. */
export function getSlashCommandDescription(
  item: SlashCommandItem,
  t: TFunction<"openhands">,
): string | null {
  const descriptionKey = (item as Partial<BuiltInSlashCommandItem>)
    .descriptionKey;
  if (descriptionKey) {
    return t(descriptionKey);
  }
  if ("description" in item.skill && item.skill.description) {
    return stripMarkdown(item.skill.description);
  }
  if ("content" in item.skill && item.skill.content) {
    return getSkillDescription(item.skill.content);
  }
  return null;
}
