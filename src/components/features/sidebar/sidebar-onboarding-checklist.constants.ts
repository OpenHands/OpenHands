import { I18nKey } from "#/i18n/declaration";
import { SCHEDULED_TASKS_INTRO_DOCS_URL } from "#/components/features/automations/scheduled-tasks-intro.constants";

export const SIDEBAR_ONBOARDING_CHECKLIST_DISMISSED_STORAGE_KEY =
  "openhands-sidebar-onboarding-checklist-dismissed";

export const SIDEBAR_ONBOARDING_CHECKLIST_DISMISSED_CHANGE_EVENT =
  "openhands-sidebar-onboarding-checklist-dismissed-change";

export const SIDEBAR_ONBOARDING_CHECKLIST_MINIMIZED_STORAGE_KEY =
  "openhands-sidebar-onboarding-checklist-minimized";

export const SIDEBAR_ONBOARDING_CHECKLIST_CUSTOMIZE_EXPLORED_STORAGE_KEY =
  "openhands-sidebar-onboarding-checklist-customize-explored";

export const SIDEBAR_ONBOARDING_CHECKLIST_ITEM_IDS = [
  "configure-llm",
  "start-conversation",
  "schedule-task",
  "customize-agent",
  "connect-mcp",
] as const;

export type SidebarOnboardingChecklistItemId =
  (typeof SIDEBAR_ONBOARDING_CHECKLIST_ITEM_IDS)[number];

export const SIDEBAR_ONBOARDING_CHECKLIST_ROUTES: Record<
  SidebarOnboardingChecklistItemId,
  string
> = {
  "configure-llm": "/settings/llm",
  "connect-mcp": "/mcp",
  "start-conversation": "/conversations",
  "schedule-task": "/automations",
  "customize-agent": "/customize",
};

export const SIDEBAR_ONBOARDING_CHECKLIST_I18N_KEYS: Record<
  SidebarOnboardingChecklistItemId,
  I18nKey
> = {
  "configure-llm": I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_CONFIGURE_LLM,
  "connect-mcp": I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_CONNECT_MCP,
  "start-conversation": I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_START_CHAT,
  "schedule-task": I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_SCHEDULE_TASK,
  "customize-agent": I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_CUSTOMIZE,
};

export const SIDEBAR_ONBOARDING_CHECKLIST_DESCRIPTION_I18N_KEYS: Record<
  SidebarOnboardingChecklistItemId,
  I18nKey
> = {
  "configure-llm": I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_CONFIGURE_LLM_DESC,
  "connect-mcp": I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_CONNECT_MCP_DESC,
  "start-conversation": I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_START_CHAT_DESC,
  "schedule-task": I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_SCHEDULE_TASK_DESC,
  "customize-agent": I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_CUSTOMIZE_DESC,
};

export const SIDEBAR_ONBOARDING_CHECKLIST_ACTION_I18N_KEYS: Record<
  SidebarOnboardingChecklistItemId,
  I18nKey
> = {
  "configure-llm": I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_ACTION_CONFIGURE_LLM,
  "connect-mcp": I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_ACTION_CONNECT_MCP,
  "start-conversation": I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_ACTION_START_CHAT,
  "schedule-task": I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_ACTION_SCHEDULE_TASK,
  "customize-agent": I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_ACTION_CUSTOMIZE,
};

export const SIDEBAR_ONBOARDING_CHECKLIST_DOCS_URLS: Record<
  SidebarOnboardingChecklistItemId,
  string
> = {
  "configure-llm":
    "https://docs.openhands.dev/openhands/usage/settings/llm-settings#llm-profiles",
  "start-conversation":
    "https://docs.openhands.dev/openhands/usage/agent-canvas/backends",
  "schedule-task": SCHEDULED_TASKS_INTRO_DOCS_URL,
  "customize-agent":
    "https://docs.openhands.dev/openhands/usage/agent-canvas/customize-and-settings",
  "connect-mcp": "https://docs.openhands.dev/overview/model-context-protocol",
};

export function isCustomizeChecklistPath(path: string): boolean {
  return (
    path === "/customize" ||
    path.startsWith("/skills") ||
    path === "/mcp" ||
    path === "/plugins"
  );
}
