import { describe, expect, it } from "vitest";
import { assessCapabilityRequirements } from "#/manifests/manifest-capabilities";
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
