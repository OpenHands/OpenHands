import { describe, expect, it } from "vitest";
import capabilitiesFixture from "@openhands/extensions/testing/automations/capabilities.json";
import incidentFixture from "@openhands/extensions/testing/automations/incident-retrospective-drafter.json";
import prReviewerFixture from "@openhands/extensions/testing/automations/github-pr-reviewer.json";
import repoMonitorFixture from "@openhands/extensions/testing/automations/github-repo-monitor.json";
import { assessCapabilityRequirements } from "#/manifests/manifest-capabilities";
import { SETUP_REGISTRY } from "#/manifests/manifest-sources";
import type { DeploymentCapabilities } from "#/manifests/types";
import { createSetup, createSetupEntry } from "./manifest-test-data";

const READY_DEPLOYMENT: DeploymentCapabilities = {
  ready: true,
  triggerKinds: ["cron"],
  eventSources: [],
  eventTypes: [],
  triggers: { cron: { minIntervalSeconds: 60, timezones: ["UTC"] } },
  features: ["repoClone", "presetPrompt"],
};

describe("assessCapabilityRequirements", () => {
  it("supports a manifest whose every requirement the deployment reports", () => {
    // Arrange
    const entry = createSetupEntry({
      requires: {
        integrations: { github: { message: "Used to read widgets." } },
        features: ["repoClone"],
      },
    });

    // Act
    const assessment = assessCapabilityRequirements(entry, READY_DEPLOYMENT);

    // Assert
    expect(assessment).toEqual({ supported: true, unmet: [] });
  });

  it("names every requirement the deployment left unreported", () => {
    // Arrange — the `cronOnly` deployment shape: reachable and ready, but no
    // webhooks can be delivered to it, so it offers neither the feature nor
    // the trigger kind an event-driven entry needs.
    const entry = createSetupEntry({
      requires: {
        integrations: { github: { message: "Used to read widgets." } },
        features: ["repoClone", "webhookDelivery"],
      },
      setup: createSetup({
        form: {
          triggers: {
            event: {
              on: {
                type: "select",
                label: "Respond to",
                help: "Which event.",
                required: true,
                options: [{ value: "push", label: "Push" }],
              },
            },
          },
          args: {},
        },
      }),
    });

    // Act
    const assessment = assessCapabilityRequirements(entry, READY_DEPLOYMENT);

    // Assert — both names, so the block can say what is missing.
    expect(assessment).toEqual({
      supported: false,
      unmet: ["webhookDelivery", "event"],
    });
  });

  it("does not support any manifest while the deployment reports it is not ready", () => {
    // Act
    const assessment = assessCapabilityRequirements(createSetupEntry(), {
      ...READY_DEPLOYMENT,
      ready: false,
    });

    // Assert — no single requirement is the reason, so none is named.
    expect(assessment).toEqual({ supported: false, unmet: [] });
  });
});

describe("the published contract fixtures", () => {
  // Each fixture bundle lists in `blockedBy` exactly the published deployment
  // shapes its entry cannot run under. The host's assessment must agree with
  // that record for every entry × shape pair.
  const shapes = Object.entries(capabilitiesFixture.responses).map(
    ([shapeName, response]) => ({
      shapeName,
      capabilities: response.body as DeploymentCapabilities,
    }),
  );

  const bundles = [prReviewerFixture, repoMonitorFixture, incidentFixture] as {
    automationId: string;
    blockedBy: string[];
  }[];

  it.each(
    bundles.flatMap(({ automationId, blockedBy }) =>
      shapes.map(({ shapeName, capabilities }) => ({
        automationId,
        shapeName,
        capabilities,
        blocked: blockedBy.includes(shapeName),
      })),
    ),
  )(
    "$automationId under the $shapeName deployment: blocked=$blocked",
    ({ automationId, capabilities, blocked }) => {
      // Arrange
      const entry = SETUP_REGISTRY.findById(automationId);
      if (!entry) throw new Error(`The registry did not admit ${automationId}`);

      // Act
      const assessment = assessCapabilityRequirements(entry, capabilities);

      // Assert
      expect(assessment.supported).toBe(!blocked);
    },
  );
});
