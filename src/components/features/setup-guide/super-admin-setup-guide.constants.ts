import type { CloudSetupGuideSteps } from "#/api/cloud/types";
import { I18nKey } from "#/i18n/declaration";
import { automationListPath } from "#/manifests/automation-interface";

/** Server permission held by instance Super Admins (`/me` `permissions`). */
export const MANAGE_SUPER_ADMINS_PERMISSION = "manage_super_admins";

/** The enterprise setup guide page, on the cloud host. */
export const SUPER_ADMIN_SETUP_GUIDE_PAGE_PATH = "/super-admin/setup";

export type SuperAdminSetupStepId =
  | "create-org"
  | "add-llm"
  | "add-integration"
  | "first-automation"
  | "invite-users"
  | "optional-saml";

/**
 * What marks a step done: a server-derived flag from `guide_steps`,
 * "guide-org" once the guide has its organization, or "optional" for a link
 * that never counts toward progress.
 */
export type SuperAdminSetupStepCompletion =
  | keyof CloudSetupGuideSteps
  | "guide-org"
  | "optional";

/**
 * Where a step opens: a page on the cloud host (`withOrg` adds the guide's
 * organization so enterprise settings open on it) or a Canvas route.
 */
export type SuperAdminSetupStepDestination =
  | { kind: "cloud"; path: string; withOrg: boolean }
  | { kind: "canvas"; path: string };

export interface SuperAdminSetupStep {
  id: SuperAdminSetupStepId;
  labelKey: I18nKey;
  completion: SuperAdminSetupStepCompletion;
  destination: SuperAdminSetupStepDestination;
}

/**
 * Mirrors the enterprise guide's steps, order and destinations so both apps
 * show the same progress.
 */
export const SUPER_ADMIN_SETUP_STEPS: readonly SuperAdminSetupStep[] = [
  {
    id: "create-org",
    labelKey: I18nKey.ONBOARDING$SETUP_GUIDE_STEP_ORG,
    completion: "guide-org",
    destination: {
      kind: "cloud",
      path: "/super-admin/organizations",
      withOrg: false,
    },
  },
  {
    id: "add-llm",
    labelKey: I18nKey.ONBOARDING$SETUP_GUIDE_STEP_LLM,
    completion: "org_llm",
    destination: {
      kind: "cloud",
      path: "/settings/org-defaults",
      withOrg: true,
    },
  },
  {
    id: "add-integration",
    labelKey: I18nKey.ONBOARDING$SETUP_GUIDE_STEP_INTEGRATION,
    completion: "mcp_server",
    destination: { kind: "cloud", path: "/settings/mcp", withOrg: true },
  },
  {
    id: "first-automation",
    labelKey: I18nKey.ONBOARDING$SETUP_GUIDE_STEP_AUTOMATION,
    completion: "automation",
    destination: { kind: "canvas", path: automationListPath() },
  },
  {
    id: "invite-users",
    labelKey: I18nKey.ONBOARDING$SETUP_GUIDE_STEP_INVITE,
    completion: "invite",
    destination: {
      kind: "cloud",
      path: "/settings/org-members",
      withOrg: true,
    },
  },
  {
    id: "optional-saml",
    labelKey: I18nKey.ONBOARDING$SETUP_GUIDE_STEP_SAML,
    completion: "optional",
    destination: {
      kind: "cloud",
      path: "/super-admin/instance",
      withOrg: false,
    },
  },
];
