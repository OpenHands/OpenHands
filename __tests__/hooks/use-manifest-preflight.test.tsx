import { act, renderHook } from "@testing-library/react";
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
  return result.current.runPreflight(VALUES);
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
  ])("returns an advisory unsupported verdict for %s", async (_case, error) => {
    // Act
    const result = await runPreflightAgainst(error);

    // Assert — an absent endpoint is the one failure mode old deployments may
    // treat as advisory.
    expect(result).toEqual({ status: "unsupported" });
    expect(warn).not.toHaveBeenCalled();
  });

  it.each([
    ["a validator that is failing", httpFailure(500)],
    ["a rejected session", axiosFailure(401)],
    ["a request that never got a response", new Error("Network Error")],
  ])("blocks on %s without logging its raw details", async (_case, error) => {
    // Act
    const result = await runPreflightAgainst(error);

    // Assert — real validator failures are not mistaken for a passing or
    // unsupported check, and provider/internal details never reach the log.
    expect(result).toEqual({ status: "unavailable" });
    expect(warn).not.toHaveBeenCalled();
  });

  it("returns every mapped field and step error from an invalid draft", async () => {
    // Arrange
    vi.mocked(AutomationService.validateDraft).mockResolvedValue({
      valid: false,
      errors: [
        {
          field: "trigger.schedule",
          code: "interval_too_short",
          message: "Choose a schedule of at least five minutes.",
        },
        {
          field: null,
          step: "prerequisites",
          code: "integration_unavailable",
          message: "Reconnect GitHub before continuing.",
        },
      ],
    });
    const { result } = renderHook(() => useSetupPreflight(ENTRY));

    // Act
    const outcome = await result.current.runPreflight(VALUES);

    // Assert
    expect(outcome).toEqual({
      status: "failed",
      errors: {
        fieldErrors: {
          schedule: "Choose a schedule of at least five minutes.",
        },
        formErrors: [],
        stepErrors: {
          prerequisites: ["Reconnect GitHub before continuing."],
        },
      },
    });
  });

  it("does not let an older response overwrite a newer verdict", async () => {
    // Arrange
    let resolveFirst!: (value: { valid: false; errors: never[] }) => void;
    let resolveSecond!: (value: { valid: true; errors: never[] }) => void;
    vi.mocked(AutomationService.validateDraft)
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveFirst = resolve;
          }),
      )
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveSecond = resolve;
          }),
      );
    const { result } = renderHook(() => useSetupPreflight(ENTRY));

    // Act — the second request completes before the first.
    const first = result.current.runPreflight(VALUES);
    const second = result.current.runPreflight({ ...VALUES, widgetName: "New" });
    resolveSecond({ valid: true, errors: [] });
    await expect(second).resolves.toEqual({ status: "passed" });
    resolveFirst({ valid: false, errors: [] });

    // Assert
    await expect(first).resolves.toEqual({ status: "stale" });
  });

  it("marks an in-flight response stale when form edits invalidate it", async () => {
    // Arrange
    let resolve!: (value: { valid: true; errors: never[] }) => void;
    vi.mocked(AutomationService.validateDraft).mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    const { result } = renderHook(() => useSetupPreflight(ENTRY));
    const pending = result.current.runPreflight(VALUES);

    // Act
    act(() => result.current.invalidatePreflight());
    resolve({ valid: true, errors: [] });

    // Assert
    await expect(pending).resolves.toEqual({ status: "stale" });
  });
});
