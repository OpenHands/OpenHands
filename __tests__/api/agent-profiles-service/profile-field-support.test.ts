import { describe, expect, it, vi } from "vitest";
import {
  agentProfileSupportsSecretRefs,
  agentProfileSupportsTools,
} from "#/api/agent-profiles-service/profile-field-support";

const mockBackendKind = vi.fn<() => string>(() => "local");

vi.mock("#/api/backend-registry/active-store", () => ({
  getActiveBackend: () => ({ backend: { kind: mockBackendKind() } }),
}));

describe.each([
  ["agentProfileSupportsSecretRefs", agentProfileSupportsSecretRefs],
  ["agentProfileSupportsTools", agentProfileSupportsTools],
])("%s", (_name, supports) => {
  it("is offered on a local backend", () => {
    mockBackendKind.mockReturnValue("local");
    expect(supports()).toBe(true);
  });

  it("stays off on Cloud, whose launches do not apply it yet", () => {
    mockBackendKind.mockReturnValue("cloud");
    expect(supports()).toBe(false);
  });
});
