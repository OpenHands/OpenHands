import { describe, expect, it } from "vitest";
import { evaluateCapabilityRequirements } from "#/manifests/manifest-capabilities";

const READY_DEPLOYMENT = {
  ready: true,
  triggerKinds: ["cron"],
  eventSources: [],
  features: ["repoClone", "presetPrompt"],
};

describe("evaluateCapabilityRequirements", () => {
  it("supports a manifest whose every requirement the deployment reports", () => {
    // Act
    const supported = evaluateCapabilityRequirements(
      { ready: true, triggerKinds: ["cron"], features: ["repoClone"] },
      READY_DEPLOYMENT,
    );

    // Assert
    expect(supported).toBe(true);
  });

  it("does not support a manifest needing something the deployment omits", () => {
    // Act
    const supported = evaluateCapabilityRequirements(
      { triggerKinds: ["event"] },
      READY_DEPLOYMENT,
    );

    // Assert
    expect(supported).toBe(false);
  });

  it("does not support any manifest while the deployment reports it is not ready", () => {
    // Act
    const supported = evaluateCapabilityRequirements(
      { ready: true },
      { ...READY_DEPLOYMENT, ready: false },
    );

    // Assert
    expect(supported).toBe(false);
  });
});
