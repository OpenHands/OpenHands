import { I18nKey } from "#/i18n/declaration";
import type {
  OnboardingLinkDestinationType,
  OnboardingLinkId,
} from "#/hooks/use-tracking";

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
  "customize-agent": "/settings/agents",
};

/** Semantic `link_id` values for `onboarding_link_clicked` (no `open_docs`). */
export const SIDEBAR_ONBOARDING_CHECKLIST_LINK_IDS: Record<
  SidebarOnboardingChecklistItemId,
  Exclude<OnboardingLinkId, "open_docs">
> = {
  "configure-llm": "configure_llm",
  "start-conversation": "start_conversation",
  "schedule-task": "schedule_task",
  "customize-agent": "customize_agent",
  "connect-mcp": "connect_mcp",
};

export const SIDEBAR_ONBOARDING_CHECKLIST_DESTINATION_TYPES: Record<
  SidebarOnboardingChecklistItemId,
  OnboardingLinkDestinationType
> = {
  "configure-llm": "settings",
  "start-conversation": "conversation",
  "schedule-task": "automation",
  "customize-agent": "settings",
  "connect-mcp": "integration",
};

export function getSidebarOnboardingChecklistHref(
  id: SidebarOnboardingChecklistItemId,
): { kind: "internal"; href: string } {
  return { kind: "internal", href: SIDEBAR_ONBOARDING_CHECKLIST_ROUTES[id] };
}

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

export function isCustomizeChecklistPath(path: string): boolean {
  return path === "/settings/agents" || path.startsWith("/settings/agents/");
}
