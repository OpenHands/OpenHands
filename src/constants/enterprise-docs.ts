import { I18nKey } from "#/i18n/declaration";

const DOCS_BASE_URL = "https://docs.openhands.dev";

/** Official docs pages the Enterprise callouts link to. */
export const ENTERPRISE_DOCS_URLS = {
  editionComparison: `${DOCS_BASE_URL}/enterprise/enterprise-vs-oss`,
  integrations: `${DOCS_BASE_URL}/enterprise/integrations/overview`,
  automations: `${DOCS_BASE_URL}/enterprise/automations`,
  llmGateways: `${DOCS_BASE_URL}/enterprise/integrations/external-llm-gateways`,
  customSandboxImages: `${DOCS_BASE_URL}/enterprise/custom-sandbox-images/index`,
  pluginMarketplace: `${DOCS_BASE_URL}/enterprise/plugin-marketplace`,
} as const;

/** Stable identifier for one callout; also the telemetry `cta_id`. */
export type EnterpriseCalloutId =
  | "compare_options"
  | "integrations"
  | "org_automations"
  | "llm_gateway"
  | "custom_sandboxes"
  | "plugin_marketplace"
  | "event_automations";

/** Where a callout is rendered; also the telemetry `surface`. */
export type EnterpriseCalloutSurface =
  | "backend_chooser"
  | "mcp_page"
  | "responder_modal"
  | "llm_settings"
  | "runtime_settings"
  | "plugins_page"
  | "automation_events";

export interface EnterpriseCalloutDefinition {
  bodyKey: I18nKey;
  linkKey: I18nKey;
  href: string;
}

export const ENTERPRISE_CALLOUTS: Record<
  EnterpriseCalloutId,
  EnterpriseCalloutDefinition
> = {
  compare_options: {
    bodyKey: I18nKey.ENTERPRISE_CTA$COMPARE_OPTIONS_BODY,
    linkKey: I18nKey.ENTERPRISE_CTA$COMPARE_OPTIONS_LINK,
    href: ENTERPRISE_DOCS_URLS.editionComparison,
  },
  integrations: {
    bodyKey: I18nKey.ENTERPRISE_CTA$INTEGRATIONS_BODY,
    linkKey: I18nKey.ENTERPRISE_CTA$INTEGRATIONS_LINK,
    href: ENTERPRISE_DOCS_URLS.integrations,
  },
  org_automations: {
    bodyKey: I18nKey.ENTERPRISE_CTA$ORG_AUTOMATIONS_BODY,
    linkKey: I18nKey.ENTERPRISE_CTA$ORG_AUTOMATIONS_LINK,
    href: ENTERPRISE_DOCS_URLS.automations,
  },
  llm_gateway: {
    bodyKey: I18nKey.ENTERPRISE_CTA$LLM_GATEWAY_BODY,
    linkKey: I18nKey.ENTERPRISE_CTA$LLM_GATEWAY_LINK,
    href: ENTERPRISE_DOCS_URLS.llmGateways,
  },
  custom_sandboxes: {
    bodyKey: I18nKey.ENTERPRISE_CTA$CUSTOM_SANDBOXES_BODY,
    linkKey: I18nKey.ENTERPRISE_CTA$CUSTOM_SANDBOXES_LINK,
    href: ENTERPRISE_DOCS_URLS.customSandboxImages,
  },
  plugin_marketplace: {
    bodyKey: I18nKey.ENTERPRISE_CTA$PLUGIN_MARKETPLACE_BODY,
    linkKey: I18nKey.ENTERPRISE_CTA$PLUGIN_MARKETPLACE_LINK,
    href: ENTERPRISE_DOCS_URLS.pluginMarketplace,
  },
  event_automations: {
    bodyKey: I18nKey.ENTERPRISE_CTA$EVENT_AUTOMATIONS_BODY,
    linkKey: I18nKey.ENTERPRISE_CTA$EVENT_AUTOMATIONS_LINK,
    href: ENTERPRISE_DOCS_URLS.automations,
  },
};
