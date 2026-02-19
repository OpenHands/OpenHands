import { I18nKey } from "#/i18n/declaration";

export interface SlashCommand {
  name: string;
  description: I18nKey;
  shortcut?: string;
  action: string;
}

export const SLASH_COMMANDS: SlashCommand[] = [
  {
    name: "/clear",
    description: I18nKey.SLASH_COMMANDS$CLEAR_DESCRIPTION,
    action: "clear",
  },
  {
    name: "/settings",
    description: I18nKey.SLASH_COMMANDS$SETTINGS_DESCRIPTION,
    shortcut: "⌘,",
    action: "settings",
  },
  {
    name: "/model",
    description: I18nKey.SLASH_COMMANDS$MODEL_DESCRIPTION,
    action: "model",
  },
];

export function filterCommands(query: string): SlashCommand[] {
  if (!query.startsWith("/")) return [];
  const searchTerm = query.toLowerCase();
  return SLASH_COMMANDS.filter((cmd) =>
    cmd.name.toLowerCase().startsWith(searchTerm),
  );
}
