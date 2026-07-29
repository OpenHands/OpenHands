import { describe, expect, it } from "vitest";
import { evaluateCapabilityRequirements } from "#/manifests/manifest-capabilities";
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

describe("evaluateCapabilityRequirements", () => {
  it("supports a manifest whose every requirement the deployment reports", () => {
    // Arrange
    const entry = createSetupEntry({
      requires: {
        integrations: { github: { message: "Used to read widgets." } },
        features: ["repoClone"],
      },
    });

    // Act
    const supported = evaluateCapabilityRequirements(entry, READY_DEPLOYMENT);

    // Assert
    expect(supported).toBe(true);
  });

  it("does not support a manifest whose trigger kind the deployment omits", () => {
    // Arrange — the form asks for an event trigger this deployment cannot fire.
    const entry = createSetupEntry({
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
          args: {
            repository: {
              type: "repo-picker",
              label: "Repository",
              help: "Which repository.",
              provider: "github",
              required: true,
            },
          },
        },
      }),
    });

    // Act
    const supported = evaluateCapabilityRequirements(entry, READY_DEPLOYMENT);

    // Assert
    expect(supported).toBe(false);
  });

  it("does not support any manifest while the deployment reports it is not ready", () => {
    // Act
    const supported = evaluateCapabilityRequirements(createSetupEntry(), {
      ...READY_DEPLOYMENT,
      ready: false,
    });

    // Assert
    expect(supported).toBe(false);
  });
});
