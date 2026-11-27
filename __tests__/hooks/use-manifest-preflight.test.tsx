import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useSetupPreflight } from "#/hooks/use-manifest-preflight";
import { createSetupEntry } from "../manifests/manifest-test-data";

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: { validateDraft: vi.fn() },
}));

/** A local call's failure, which arrives as an `AxiosError`. */
function axiosFailure(status: number) {
  return Object.assign(new Error(`Request failed with status ${status}`), {
    isAxiosError: true,
    response: { status },
  });
}

/** A cloud call's failure, which arrives as the shared client's `HttpError`. */
function httpFailure(status: number) {
  return Object.assign(new Error(`Request failed with status ${status}`), {
    name: "HttpError",
    status,
  });
}

const ENTRY = createSetupEntry();
const VALUES = { repository: "OpenHands/agent-server-gui", widgetName: "W" };

function runPreflightAgainst(error: unknown) {
  vi.mocked(AutomationService.validateDraft).mockRejectedValue(error);
  const { result } = renderHook(() => useSetupPreflight(ENTRY));
  return result.current(VALUES);
}

let warn: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  vi.clearAllMocks();
  warn = vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  warn.mockRestore();
});

describe("useSetupPreflight", () => {
  it.each([
    ["a local deployment without the route", axiosFailure(404)],
    ["a cloud deployment that has not implemented it", httpFailure(501)],
  ])("has no verdict and nothing to report for %s", async (_case, error) => {
    // Act
    const result = await runPreflightAgainst(error);

    // Assert — preflight is advisory, and an absent endpoint is an expected
    // state rather than something worth reporting.
    expect(result).toBeNull();
    expect(warn).not.toHaveBeenCalled();
  });

  it.each([
    ["a validator that is failing", httpFailure(500)],
    ["a rejected session", axiosFailure(401)],
    ["a request that never got a response", new Error("Network Error")],
  ])("reports %s while still letting setup continue", async (_case, error) => {
    // Act
    const result = await runPreflightAgainst(error);

    // Assert — the user is not blocked, but the failure is not invisible
    // either; without this it reads exactly like an unimplemented endpoint.
    expect(result).toBeNull();
    expect(warn).toHaveBeenCalledWith(
      "Automation setup preflight failed:",
      error,
    );
  });
});
