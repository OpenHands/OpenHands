import { I18nKey } from "#/i18n/declaration";
import {
  SUPER_ADMIN_SETUP_STEPS,
  type SuperAdminSetupStepId,
} from "../super-admin-setup-guide.constants";
import { startGuidedTour, type GuidedTourNavigation } from "./tour-engine";
import type { GuidedTourStop } from "./tour-types";

type TourStopSpec = Omit<GuidedTourStop, "title" | "body"> & {
  title: I18nKey;
  body: I18nKey;
};

/**
 * Tour stops for the setup steps done in Canvas. Each step's first stop is on
 * the step's page; the next one follows the admin's click.
 */
const SETUP_GUIDE_TOUR_STOPS: Partial<
  Record<SuperAdminSetupStepId, TourStopSpec[]>
> = {
  "first-automation": [
    {
      id: "choose-template",
      anchor: '[data-testid="recommended-automations-section"]',
      advanceOnClick: '[data-testid^="recommended-automation-card-"]',
      // Next is hidden while it waits for a card, and Back has nowhere to go.
      hideFooter: true,
      title: I18nKey.ONBOARDING$SETUP_GUIDE_TOUR_TEMPLATE_TITLE,
      body: I18nKey.ONBOARDING$SETUP_GUIDE_TOUR_TEMPLATE_BODY,
      side: "top",
    },
    {
      id: "set-up-automation",
      anchor: '[data-testid="setup-dialog"]',
      waitForComplete: "first-automation",
      interactive: true,
      title: I18nKey.ONBOARDING$SETUP_GUIDE_TOUR_AUTOMATION_TITLE,
      body: I18nKey.ONBOARDING$SETUP_GUIDE_TOUR_AUTOMATION_BODY,
      side: "left",
    },
  ],
  "add-integration": [
    {
      id: "add-custom-server",
      anchor: '[data-testid="mcp-add-custom-server"]',
      nextClicksAnchor: true,
      title: I18nKey.ONBOARDING$SETUP_GUIDE_TOUR_MCP_ADD_TITLE,
      body: I18nKey.ONBOARDING$SETUP_GUIDE_TOUR_MCP_ADD_BODY,
      side: "bottom",
    },
    {
      id: "connect-server",
      anchor: '[data-testid="mcp-custom-editor"]',
      waitForComplete: "add-integration",
      interactive: true,
      title: I18nKey.ONBOARDING$SETUP_GUIDE_TOUR_MCP_FORM_TITLE,
      body: I18nKey.ONBOARDING$SETUP_GUIDE_TOUR_MCP_FORM_BODY,
      side: "left",
    },
  ],
};

export function hasSetupGuideTour(stepId: SuperAdminSetupStepId): boolean {
  return Boolean(SETUP_GUIDE_TOUR_STOPS[stepId]?.length);
}

/** Open a Canvas setup step's page and walk the admin through it. */
export async function startSetupGuideTour(
  stepId: SuperAdminSetupStepId,
  nav: GuidedTourNavigation,
  t: (key: I18nKey) => string,
): Promise<void> {
  const specs = SETUP_GUIDE_TOUR_STOPS[stepId];
  const step = SUPER_ADMIN_SETUP_STEPS.find(({ id }) => id === stepId);
  if (!specs || step?.destination.kind !== "canvas") {
    return;
  }
  const route = step.destination.path;
  const stops = specs.map(({ title, body, ...stop }, index) => ({
    ...stop,
    route: index === 0 ? route : undefined,
    title: t(title),
    body: t(body),
  }));
  await startGuidedTour({ id: stepId, stops }, nav, {
    next: t(I18nKey.ONBOARDING$NEXT),
    previous: t(I18nKey.ONBOARDING$BACK),
    done: t(I18nKey.ONBOARDING$SETUP_GUIDE_TOUR_DONE),
  });
}
