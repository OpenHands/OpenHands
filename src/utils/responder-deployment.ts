import type { RecommendedAutomation } from "@openhands/extensions/automations";
import { I18nKey } from "#/i18n/declaration";
import { PRODUCT_URL } from "#/utils/constants";

/** OpenHands Cloud integrations page — where always-on responders are set up. */
export const OPENHANDS_CLOUD_INTEGRATIONS_URL = `${PRODUCT_URL.PRODUCTION}/settings/integrations`;

/**
 * Integrations whose automations are treated as event "responders" that poll
 * continuously. A responder is an automation whose required integrations are
 * exclusively from this set, so multi-tool digests (e.g. slack + linear +
 * notion) are not treated as Slack responders.
 */
const RESPONDER_INTEGRATION_IDS = ["github", "slack"];

/**
 * Single source of truth for "does this automation get the deployment-choice
 * modal?" — true only for pure GitHub/Slack responders.
 */
export function isResponderAutomation(
  automation: RecommendedAutomation,
): boolean {
  const ids = automation.requiredIntegrationIds;
  return (
    ids.length > 0 && ids.every((id) => RESPONDER_INTEGRATION_IDS.includes(id))
  );
}

/**
 * Where a responder runs. Only `local` and `openhands-cloud` are wired today;
 * `user-cloud` is reserved for a future remote-deployment target.
 */
export type ResponderDeploymentTarget =
  | "local"
  | "user-cloud"
  | "openhands-cloud";

/** What the launcher should do when an option's primary action fires. */
export type ResponderDeploymentAction =
  | { kind: "launch-local" }
  | { kind: "open-url"; url: string };

/** Presentational + behavioral descriptor for one deployment option. */
export interface ResponderDeploymentOption {
  target: ResponderDeploymentTarget;
  testId: string;
  titleKey: I18nKey;
  descriptionKey: I18nKey;
  primaryActionKey: I18nKey;
  primaryActionTestId: string;
  action: ResponderDeploymentAction;
}

/**
 * Centralized runtime-selection mechanism. Returns a descriptor (data) rather
 * than executing behavior, so the side effects stay with the launcher that owns
 * the relevant state. Adding a target later means adding a `case` here and an
 * entry to {@link VISIBLE_RESPONDER_DEPLOYMENT_TARGETS}; the `never` check makes
 * an unhandled target a compile error.
 */
export function resolveResponderDeploymentOption(
  target: ResponderDeploymentTarget,
): ResponderDeploymentOption {
  switch (target) {
    case "local":
      return {
        target,
        testId: "responder-deployment-option-local",
        titleKey: I18nKey.RESPONDER_DEPLOYMENT$LOCAL_TITLE,
        descriptionKey: I18nKey.RESPONDER_DEPLOYMENT$LOCAL_DESCRIPTION,
        primaryActionKey: I18nKey.RESPONDER_DEPLOYMENT$LOCAL_ACTION,
        primaryActionTestId: "responder-deployment-continue-local",
        action: { kind: "launch-local" },
      };
    case "openhands-cloud":
      return {
        target,
        testId: "responder-deployment-option-openhands-cloud",
        titleKey: I18nKey.RESPONDER_DEPLOYMENT$OPENHANDS_CLOUD_TITLE,
        descriptionKey:
          I18nKey.RESPONDER_DEPLOYMENT$OPENHANDS_CLOUD_DESCRIPTION,
        primaryActionKey: I18nKey.RESPONDER_DEPLOYMENT$OPENHANDS_CLOUD_ACTION,
        primaryActionTestId: "responder-deployment-open-openhands-cloud",
        action: { kind: "open-url", url: OPENHANDS_CLOUD_INTEGRATIONS_URL },
      };
    case "user-cloud":
      throw new Error("User Cloud responder deployment is not yet supported");
    default: {
      const exhaustive: never = target;
      return exhaustive;
    }
  }
}

/** Targets the modal renders today, in display order (excludes `user-cloud`). */
export const VISIBLE_RESPONDER_DEPLOYMENT_TARGETS: ResponderDeploymentTarget[] =
  ["local", "openhands-cloud"];
