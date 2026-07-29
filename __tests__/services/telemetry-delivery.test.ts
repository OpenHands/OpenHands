import { renderHook, waitFor } from "@testing-library/react";
import posthog from "posthog-js";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  clearTelemetryData,
  initializePostHogClient,
  setTelemetryConsent,
  trackInstall,
} from "#/services/telemetry";
import { useTracking } from "#/hooks/use-tracking";

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({ data: undefined }),
}));

describe("Canvas telemetry delivery", () => {
  afterEach(async () => {
    await clearTelemetryData();
  });

  it("keeps install and backend-transition events on one client identity", async () => {
    const client = await initializePostHogClient();
    expect(client).not.toBeNull();
    expect(client).not.toBe(posthog);
    const capture = vi.spyOn(client!, "capture");

    await trackInstall();
    expect(capture).toHaveBeenCalledWith("canvas_install", expect.any(Object));
    const installDistinctId = client!.get_distinct_id();

    await setTelemetryConsent("granted");
    const { result } = renderHook(() => useTracking());

    result.current.trackBackendAdded({
      backendKind: "cloud",
      connectionMethod: "cloud_login",
      hasApiKey: true,
      source: "onboarding",
    });
    await waitFor(() =>
      expect(capture).toHaveBeenCalledWith("backend_added", expect.any(Object)),
    );
    expect(client!.get_distinct_id()).toBe(installDistinctId);
  });
});
