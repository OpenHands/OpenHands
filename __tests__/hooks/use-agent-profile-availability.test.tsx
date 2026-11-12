import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useAgentProfileAvailability } from "#/hooks/use-agent-profile-availability";

const { config } = vi.hoisted(() => ({ config: vi.fn() }));
vi.mock("#/hooks/query/use-cloud-integrations-config", () => ({
  useCloudIntegrationsConfig: () => config(),
}));

describe("useAgentProfileAvailability", () => {
  it.each([
    ["local or unknown backend", undefined, true],
    [
      "Cloud without the self-hosted flag",
      { deploymentMode: "cloud", isAgentProfilesEnabled: false },
      true,
    ],
    [
      "self-hosted profiles enabled",
      { deploymentMode: "self_hosted", isAgentProfilesEnabled: true },
      true,
    ],
    [
      "self-hosted profiles disabled",
      { deploymentMode: "self_hosted", isAgentProfilesEnabled: false },
      false,
    ],
  ])("preserves selection policy for %s", (_name, data, expected) => {
    config.mockReturnValue({ data });
    const { result } = renderHook(useAgentProfileAvailability);
    expect(result.current).toBe(expected);
  });

  it("recomputes when the connected backend configuration changes", () => {
    config.mockReturnValue({
      data: { deploymentMode: "self_hosted", isAgentProfilesEnabled: false },
    });
    const { result, rerender } = renderHook(useAgentProfileAvailability);
    expect(result.current).toBe(false);
    config.mockReturnValue({ data: undefined, isError: true });
    rerender();
    expect(result.current).toBe(true);
  });
});
