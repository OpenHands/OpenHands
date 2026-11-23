import { afterEach, describe, expect, it } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import {
  agentProfileSupportsInstructions,
  agentProfileSupportsRouter,
  agentProfileSupportsSecretRefs,
  agentProfileSupportsSkills,
  agentProfileSupportsTools,
} from "#/api/agent-profiles-service/profile-field-support";

afterEach(() => {
  __resetActiveStoreForTests();
});

describe.each([
  ["agentProfileSupportsSecretRefs", agentProfileSupportsSecretRefs],
  ["agentProfileSupportsInstructions", agentProfileSupportsInstructions],
  ["agentProfileSupportsTools", agentProfileSupportsTools],
  ["agentProfileSupportsSkills", agentProfileSupportsSkills],
  ["agentProfileSupportsRouter", agentProfileSupportsRouter],
])("%s", (_name, supports) => {
  it("is offered on a local backend", () => {
    expect(supports()).toBe(true);
  });

  it("is offered on Cloud, whose profile launches now apply it", () => {
    setRegisteredBackends([
      {
        id: "cloud",
        name: "Cloud",
        host: "https://app.example.test",
        apiKey: "test-key",
        kind: "cloud",
      },
    ]);
    setActiveSelection({ backendId: "cloud" });
    expect(supports()).toBe(true);
  });
});
