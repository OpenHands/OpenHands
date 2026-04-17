import { useConfig } from "#/hooks/query/use-config";
import {
  SAAS_NAV_ITEMS,
  OSS_NAV_ITEMS,
  SettingsNavItem,
  SettingsNavSection,
} from "#/constants/settings-nav";
import { OrganizationUserRole } from "#/types/org";
import { isBillingHidden } from "#/utils/org/billing-visibility";
import { isSettingsPageHidden } from "#/utils/settings-utils";
import { useMe } from "./query/use-me";
import { useSettings } from "./query/use-settings";
import { usePermission } from "./organizations/use-permissions";
import { useOrgTypeAndAccess } from "./use-org-type-and-access";
import { I18nKey } from "#/i18n/declaration";

/** Settings-nav entries that only make sense for the standard LLM agent.
 *
 * When the user has picked the ACP variant (``agent_kind === "acp"``),
 * the ACP subprocess owns its own system prompt, tool set, memory,
 * skills, and MCP — so these nav entries would just route to pages
 * that silently ignore what the user saves. We hide them so the
 * sidebar reflects reality. The Agent Type page
 * (``/settings/agent-type``) is how the user switches back. */
const LLM_ONLY_NAV_PATHS = new Set<string>([
  "/settings", // LLM page — only applies to the OpenHands agent.
  "/settings/condenser",
  "/settings/verification",
  "/settings/mcp",
  "/settings/skills", // Skills get injected into LLM agent_context; ACP ignores them.
  "/settings/org-defaults",
  "/settings/org-defaults/condenser",
  "/settings/org-defaults/verification",
]);

/** Settings-nav entries that only apply to the ACP variant. Hidden
 *  when ``agent_kind !== "acp"`` so OpenHands-agent users don't see
 *  ACP-subprocess configuration they can't act on. */
const ACP_ONLY_NAV_PATHS = new Set<string>([
  "/settings/acp-server",
  "/settings/acp-model",
]);

// Rendered navigation item types
export type SettingsNavRenderedItem =
  | { type: "item"; item: SettingsNavItem }
  | { type: "header"; text: I18nKey }
  | { type: "divider" };

// Section header text mapping
const SECTION_HEADERS: Partial<Record<SettingsNavSection, I18nKey>> = {
  org: I18nKey.SETTINGS$ORG_SETTINGS_HEADER,
  personal: I18nKey.SETTINGS$PERSONAL_SETTINGS_HEADER,
};

/**
 * Build Settings navigation items based on:
 * - app mode (saas / oss)
 * - feature flags
 * - active user's role
 * - org type (personal vs team)
 * @returns Settings Nav Rendered Items (items, headers, dividers)
 */
export function useSettingsNavItems(): SettingsNavRenderedItem[] {
  const { data: config } = useConfig();
  const { data: user } = useMe();
  const { data: settings } = useSettings();
  const userRole: OrganizationUserRole = user?.role ?? "member";
  const { hasPermission } = usePermission(userRole);
  const { isPersonalOrg, isTeamOrg, organizationId } = useOrgTypeAndAccess();

  // ACP replaces the standard LLM agent wholesale; hide the LLM-only
  // sub-tabs so the sidebar doesn't suggest they still apply.
  const agentSettings = settings?.agent_settings as
    | Record<string, unknown>
    | undefined;
  const agentKind =
    typeof agentSettings?.agent_kind === "string"
      ? (agentSettings.agent_kind as string)
      : "llm";
  const hideLlmOnlyNav = agentKind === "acp";

  const shouldHideBilling = isBillingHidden(
    config,
    hasPermission("view_billing"),
  );
  const isSaasMode = config?.app_mode === "saas";
  const featureFlags = config?.feature_flags;
  const isAdminOrOwner = userRole === "admin" || userRole === "owner";

  let items = isSaasMode ? [...SAAS_NAV_ITEMS] : [...OSS_NAV_ITEMS];

  // First apply feature flag-based hiding
  items = items.filter((item) => !isSettingsPageHidden(item.to, featureFlags));

  // Reshape the sidebar to match the active agent kind:
  // - ACP mode hides LLM-only sub-tabs (LLM, Condenser, Verification,
  //   MCP, Skills) — those don't apply when the ACP subprocess owns
  //   the agent loop.
  // - LLM (OpenHands) mode hides the ACP-only sub-tabs (ACP Server,
  //   ACP Model).
  if (hideLlmOnlyNav) {
    items = items.filter((item) => !LLM_ONLY_NAV_PATHS.has(item.to));
  } else {
    items = items.filter((item) => !ACP_ONLY_NAV_PATHS.has(item.to));
  }

  // Hide billing when billing is not accessible OR when in team org
  if (shouldHideBilling || isTeamOrg) {
    items = items.filter((item) => item.to !== "/settings/billing");
  }

  // Hide org routes for personal orgs, missing permissions, or no org selected
  if (!hasPermission("view_billing") || !organizationId || isPersonalOrg) {
    items = items.filter((item) => item.to !== "/settings/org");
  }

  if (
    !hasPermission("invite_user_to_organization") ||
    !organizationId ||
    isPersonalOrg
  ) {
    items = items.filter((item) => item.to !== "/settings/org-members");
  }

  if (!hasPermission("edit_llm_settings") || !organizationId || isPersonalOrg) {
    items = items.filter(
      (item) => !item.to.startsWith("/settings/org-defaults"),
    );
  }

  // For OSS mode or non-SaaS, return flat list without sections
  if (!isSaasMode) {
    return items.map((item) => ({ type: "item", item }));
  }

  // Build rendered items with headers and dividers for SaaS mode
  const renderedItems: SettingsNavRenderedItem[] = [];
  let currentSection: SettingsNavSection | undefined;
  let isFirstSection = true;

  // Determine if we should show section headers (only for admins/owners in team orgs)
  const showSectionHeaders = isTeamOrg && isAdminOrOwner;

  for (const item of items) {
    const itemSection = item.section;

    // Check if we're entering a new section
    if (itemSection && itemSection !== currentSection) {
      // For personal orgs or members, treat "org" and "personal" sections as one group
      // (LLM is the only org item visible and should flow with personal items)
      const isOrgToPersonalWithoutHeaders =
        (isPersonalOrg || !isAdminOrOwner) &&
        currentSection === "org" &&
        itemSection === "personal";

      // Add divider between sections (but not before the first section,
      // and not between org->personal when section headers aren't shown)
      if (!isFirstSection && !isOrgToPersonalWithoutHeaders) {
        renderedItems.push({ type: "divider" });
      }

      // Add section header for org and personal sections (admins/owners only)
      if (showSectionHeaders && SECTION_HEADERS[itemSection]) {
        renderedItems.push({
          type: "header",
          text: SECTION_HEADERS[itemSection]!,
        });
      }

      currentSection = itemSection;
      isFirstSection = false;
    }

    renderedItems.push({ type: "item", item });
  }

  return renderedItems;
}
