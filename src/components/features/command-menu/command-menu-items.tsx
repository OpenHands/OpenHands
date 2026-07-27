import React from "react";
import {
  Bot,
  Home,
  Keyboard,
  KeyRound,
  ListTodo,
  PanelsTopLeft,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  Wrench,
  Zap,
} from "lucide-react";
import { I18nKey } from "#/i18n/declaration";

const ICON_SIZE = 18;

export const COMMAND_MENU_ROUTE = {
  conversations: "/conversations",
  customize: "/customize",
  automations: "/automations",
  mcp: "/mcp",
  settings: "/settings",
  agentSettings: "/settings/agents",
  llmSettings: "/settings/llm",
  condenserSettings: "/settings/condenser",
  verificationSettings: "/settings/verification",
  appSettings: "/settings/app",
  secretsSettings: "/settings/secrets",
} as const;

export type CommandMenuGroupId = "navigation" | "settings" | "actions";
export type CommandMenuItemId =
  | "new-chat"
  | "customize"
  | "automations"
  | "mcp"
  | "settings"
  | "agent-settings"
  | "llm-settings"
  | "condenser-settings"
  | "verification-settings"
  | "app-settings"
  | "secrets-settings"
  | "toggle-sidebar";

export interface CommandMenuItemDefinition {
  id: CommandMenuItemId;
  group: CommandMenuGroupId;
  titleKey: I18nKey;
  descriptionKey: I18nKey;
  keywordsKey: I18nKey;
  icon: React.ReactElement;
  to?: string;
  perform?: () => void;
}

export const COMMAND_MENU_GROUP_LABELS: Record<CommandMenuGroupId, I18nKey> = {
  navigation: I18nKey.COMMAND_MENU$GROUP_NAVIGATION,
  settings: I18nKey.COMMAND_MENU$GROUP_SETTINGS,
  actions: I18nKey.COMMAND_MENU$GROUP_ACTIONS,
};

export const COMMAND_MENU_GROUP_ORDER: CommandMenuGroupId[] = [
  "navigation",
  "settings",
  "actions",
];

export const createCommandMenuItems = ({
  toggleSidebar,
}: {
  toggleSidebar: () => void;
}): CommandMenuItemDefinition[] => [
  {
    id: "new-chat",
    group: "navigation",
    titleKey: I18nKey.COMMAND_MENU$NEW_CHAT_TITLE,
    descriptionKey: I18nKey.COMMAND_MENU$NEW_CHAT_DESCRIPTION,
    keywordsKey: I18nKey.COMMAND_MENU$NEW_CHAT_KEYWORDS,
    icon: <Home size={ICON_SIZE} />,
    to: COMMAND_MENU_ROUTE.conversations,
  },
  {
    id: "customize",
    group: "navigation",
    titleKey: I18nKey.COMMAND_MENU$CUSTOMIZE_TITLE,
    descriptionKey: I18nKey.COMMAND_MENU$CUSTOMIZE_DESCRIPTION,
    keywordsKey: I18nKey.COMMAND_MENU$CUSTOMIZE_KEYWORDS,
    icon: <Sparkles size={ICON_SIZE} />,
    to: COMMAND_MENU_ROUTE.customize,
  },
  {
    id: "automations",
    group: "navigation",
    titleKey: I18nKey.COMMAND_MENU$AUTOMATIONS_TITLE,
    descriptionKey: I18nKey.COMMAND_MENU$AUTOMATIONS_DESCRIPTION,
    keywordsKey: I18nKey.COMMAND_MENU$AUTOMATIONS_KEYWORDS,
    icon: <Zap size={ICON_SIZE} />,
    to: COMMAND_MENU_ROUTE.automations,
  },
  {
    id: "mcp",
    group: "navigation",
    titleKey: I18nKey.COMMAND_MENU$MCP_TITLE,
    descriptionKey: I18nKey.COMMAND_MENU$MCP_DESCRIPTION,
    keywordsKey: I18nKey.COMMAND_MENU$MCP_KEYWORDS,
    icon: <Wrench size={ICON_SIZE} />,
    to: COMMAND_MENU_ROUTE.mcp,
  },
  {
    id: "settings",
    group: "settings",
    titleKey: I18nKey.COMMAND_MENU$SETTINGS_TITLE,
    descriptionKey: I18nKey.COMMAND_MENU$SETTINGS_DESCRIPTION,
    keywordsKey: I18nKey.COMMAND_MENU$SETTINGS_KEYWORDS,
    icon: <Settings size={ICON_SIZE} />,
    to: COMMAND_MENU_ROUTE.settings,
  },
  {
    id: "agent-settings",
    group: "settings",
    titleKey: I18nKey.COMMAND_MENU$AGENT_SETTINGS_TITLE,
    descriptionKey: I18nKey.COMMAND_MENU$AGENT_SETTINGS_DESCRIPTION,
    keywordsKey: I18nKey.COMMAND_MENU$AGENT_SETTINGS_KEYWORDS,
    icon: <Bot size={ICON_SIZE} />,
    to: COMMAND_MENU_ROUTE.agentSettings,
  },
  {
    id: "llm-settings",
    group: "settings",
    titleKey: I18nKey.COMMAND_MENU$LLM_SETTINGS_TITLE,
    descriptionKey: I18nKey.COMMAND_MENU$LLM_SETTINGS_DESCRIPTION,
    keywordsKey: I18nKey.COMMAND_MENU$LLM_SETTINGS_KEYWORDS,
    icon: <Search size={ICON_SIZE} />,
    to: COMMAND_MENU_ROUTE.llmSettings,
  },
  {
    id: "condenser-settings",
    group: "settings",
    titleKey: I18nKey.COMMAND_MENU$CONDENSER_SETTINGS_TITLE,
    descriptionKey: I18nKey.COMMAND_MENU$CONDENSER_SETTINGS_DESCRIPTION,
    keywordsKey: I18nKey.COMMAND_MENU$CONDENSER_SETTINGS_KEYWORDS,
    icon: <ListTodo size={ICON_SIZE} />,
    to: COMMAND_MENU_ROUTE.condenserSettings,
  },
  {
    id: "verification-settings",
    group: "settings",
    titleKey: I18nKey.COMMAND_MENU$VERIFICATION_SETTINGS_TITLE,
    descriptionKey: I18nKey.COMMAND_MENU$VERIFICATION_SETTINGS_DESCRIPTION,
    keywordsKey: I18nKey.COMMAND_MENU$VERIFICATION_SETTINGS_KEYWORDS,
    icon: <ShieldCheck size={ICON_SIZE} />,
    to: COMMAND_MENU_ROUTE.verificationSettings,
  },
  {
    id: "app-settings",
    group: "settings",
    titleKey: I18nKey.COMMAND_MENU$APP_SETTINGS_TITLE,
    descriptionKey: I18nKey.COMMAND_MENU$APP_SETTINGS_DESCRIPTION,
    keywordsKey: I18nKey.COMMAND_MENU$APP_SETTINGS_KEYWORDS,
    icon: <PanelsTopLeft size={ICON_SIZE} />,
    to: COMMAND_MENU_ROUTE.appSettings,
  },
  {
    id: "secrets-settings",
    group: "settings",
    titleKey: I18nKey.COMMAND_MENU$SECRETS_SETTINGS_TITLE,
    descriptionKey: I18nKey.COMMAND_MENU$SECRETS_SETTINGS_DESCRIPTION,
    keywordsKey: I18nKey.COMMAND_MENU$SECRETS_SETTINGS_KEYWORDS,
    icon: <KeyRound size={ICON_SIZE} />,
    to: COMMAND_MENU_ROUTE.secretsSettings,
  },
  {
    id: "toggle-sidebar",
    group: "actions",
    titleKey: I18nKey.COMMAND_MENU$TOGGLE_SIDEBAR_TITLE,
    descriptionKey: I18nKey.COMMAND_MENU$TOGGLE_SIDEBAR_DESCRIPTION,
    keywordsKey: I18nKey.COMMAND_MENU$TOGGLE_SIDEBAR_KEYWORDS,
    icon: <Keyboard size={ICON_SIZE} />,
    perform: toggleSidebar,
  },
];
