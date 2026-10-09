import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  SUPER_ADMIN_SETUP_STEPS,
  type SuperAdminSetupStepId,
} from "#/components/features/setup-guide/super-admin-setup-guide.constants";
import {
  hasSetupGuideTour,
  startSetupGuideTour,
} from "#/components/features/setup-guide/tour/setup-guide-tour";
import { startGuidedTour } from "#/components/features/setup-guide/tour/tour-engine";

vi.mock("#/components/features/setup-guide/tour/tour-engine", () => ({
  startGuidedTour: vi.fn(),
}));

const t = (key: string) => `t:${key}`;
const NAV = { navigate: vi.fn(), getPath: () => "/" };

describe("setup guide tours", () => {
  beforeEach(() => {
    vi.mocked(startGuidedTour).mockReset();
  });

  it("has a tour for each step done in Canvas, and only those", () => {
    expect(
      SUPER_ADMIN_SETUP_STEPS.filter(({ id }) => hasSetupGuideTour(id)).map(
        ({ id }) => id,
      ),
    ).toEqual(["first-automation", "add-integration"]);
  });

  it.each<[SuperAdminSetupStepId, string]>([
    ["first-automation", "/automations/templates"],
    ["add-integration", "/mcp"],
  ])(
    "starts the %s tour on the step's page and waits for the step",
    async (stepId, route) => {
      await startSetupGuideTour(stepId, NAV, t);

      expect(startGuidedTour).toHaveBeenCalledTimes(1);
      const [tour, nav, labels] = vi.mocked(startGuidedTour).mock.calls[0];
      expect(nav).toBe(NAV);
      expect(tour.stops.map((stop) => stop.route)).toEqual([route, undefined]);
      expect(tour.stops.at(-1)?.waitForComplete).toBe(stepId);
      for (const stop of tour.stops) {
        expect(stop.title).toMatch(/^t:ONBOARDING\$SETUP_GUIDE_TOUR_/);
        expect(stop.body).toMatch(/^t:ONBOARDING\$SETUP_GUIDE_TOUR_/);
      }
      expect(labels).toEqual({
        next: "t:ONBOARDING$NEXT",
        previous: "t:ONBOARDING$BACK",
        done: "t:ONBOARDING$SETUP_GUIDE_TOUR_DONE",
      });
    },
  );

  it("hides the footer only on the stop that waits for a template card", async () => {
    await startSetupGuideTour("first-automation", NAV, t);
    await startSetupGuideTour("add-integration", NAV, t);

    const stops = vi
      .mocked(startGuidedTour)
      .mock.calls.flatMap(([tour]) => tour.stops);
    expect(
      stops.filter((stop) => stop.hideFooter).map((stop) => stop.id),
    ).toEqual(["choose-template"]);
  });

  it("starts nothing for a step done in the enterprise app", async () => {
    await startSetupGuideTour("invite-users", NAV, t);

    expect(startGuidedTour).not.toHaveBeenCalled();
  });
});
