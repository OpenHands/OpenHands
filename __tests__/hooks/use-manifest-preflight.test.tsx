import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import AutomationService from "#/api/automation-service/automation-service.api";
import { type ResolvedActiveBackend } from "#/api/backend-registry/types";
import { useSetupPreflight } from "#/hooks/use-manifest-preflight";
import { createSetupEntry } from "../manifests/manifest-test-data";

const activeBackendState = vi.hoisted((): { value: ResolvedActiveBackend } => ({
  value: {
    backend: {
      id: "local-test",
      name: "Local test",
      host: "http://localhost:3000",
      apiKey: "session-key",
      kind: "local",
      connectionRevision: 0,
    },
    orgId: null,
  },
}));

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: { validateDraft: vi.fn() },
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => activeBackendState.value,
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

function requirementsSchemaFailure() {
  return Object.assign(new Error("Request failed with status 422"), {
    isAxiosError: true,
    response: {
      status: 422,
      data: {
        detail: [
          {
            loc: ["body", "requirements"],
            type: "extra_forbidden",
            msg: "Extra inputs are not permitted",
            input: { secretNames: ["GITHUB_TOKEN"] },
          },
        ],
      },
    },
  });
}

function sdkRequirementsSchemaFailure() {
  return Object.assign(new Error("Request failed with status 422"), {
    name: "HttpError",
    status: 422,
    response: {
      detail: [
        {
          loc: ["body", "requirements"],
          type: "value_error.extra",
          msg: "Extra inputs are not permitted",
        },
      ],
    },
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
  vi.resetAllMocks();
  activeBackendState.value = {
    backend: {
      id: "local-test",
      name: "Local test",
      host: "http://localhost:3000",
      apiKey: "session-key",
      kind: "local",
      connectionRevision: 0,
    },
    orgId: null,
  };
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

  it.each([
    [
      "valid with errors",
      { valid: true, errors: [{ message: "contradiction" }] },
    ],
    ["invalid without errors", { valid: false, errors: [] }],
    ["wrong valid type", { valid: "yes", errors: [] }],
    ["non-array errors", { valid: true, errors: "" }],
    ["null response", null],
  ])("fails closed for a malformed %s response", async (_case, response) => {
    vi.mocked(AutomationService.validateDraft).mockResolvedValue(
      response as never,
    );
    const { result } = renderHook(() => useSetupPreflight(ENTRY));

    await expect(result.current.runPreflight(VALUES)).resolves.toEqual({
      status: "unavailable",
    });
  });

  it.each([
    ["the local HTTP client", requirementsSchemaFailure()],
    ["the cloud SDK client", sdkRequirementsSchemaFailure()],
  ])(
    "uses the legacy draft contract when %s rejects requirements as unknown",
    async (_transport, error) => {
      // Arrange
      vi.mocked(AutomationService.validateDraft)
        .mockRejectedValueOnce(error)
        .mockResolvedValueOnce({ valid: true, errors: [] });
      const { result } = renderHook(() => useSetupPreflight(ENTRY));

      // Act
      const outcome = await result.current.runPreflight(VALUES);

      // Assert - the old endpoint can still validate the draft, but it did not
      // perform the new deployment checks, so the verdict remains advisory.
      expect(outcome).toEqual({ status: "unsupported" });
      expect(AutomationService.validateDraft).toHaveBeenCalledTimes(2);
      expect(
        vi.mocked(AutomationService.validateDraft).mock.calls[1][0],
      ).not.toHaveProperty("requirements");
      expect(warn).not.toHaveBeenCalled();
    },
  );

  it("keeps legacy draft validation errors blocking after the compatibility retry", async () => {
    // Arrange
    vi.mocked(AutomationService.validateDraft)
      .mockRejectedValueOnce(requirementsSchemaFailure())
      .mockResolvedValueOnce({
        valid: false,
        errors: [
          {
            field: "trigger.schedule",
            code: "invalid_schedule",
            message: "Choose a valid schedule.",
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
        fieldErrors: { schedule: "Choose a valid schedule." },
        formErrors: [],
        stepErrors: {},
      },
    });
  });

  it("does not treat an unrelated 422 as an unsupported deployment", async () => {
    // Arrange
    vi.mocked(AutomationService.validateDraft).mockRejectedValue(
      Object.assign(new Error("Request failed with status 422"), {
        isAxiosError: true,
        response: {
          status: 422,
          data: {
            detail: [
              {
                loc: ["body", "draft", "trigger"],
                type: "value_error",
                msg: "Invalid trigger",
              },
            ],
          },
        },
      }),
    );
    const { result } = renderHook(() => useSetupPreflight(ENTRY));

    // Act
    const outcome = await result.current.runPreflight(VALUES);

    // Assert
    expect(outcome).toEqual({ status: "unavailable" });
    expect(AutomationService.validateDraft).toHaveBeenCalledTimes(1);
  });

  it("does not retry when a nested requirements value is malformed", async () => {
    // Arrange - this is a real validation error from a current endpoint, not
    // evidence that the endpoint predates the top-level requirements field.
    vi.mocked(AutomationService.validateDraft).mockRejectedValue(
      Object.assign(new Error("Request failed with status 422"), {
        isAxiosError: true,
        response: {
          status: 422,
          data: {
            detail: [
              {
                loc: ["body", "requirements", "integrations", 0, "id"],
                type: "extra_forbidden",
                msg: "Extra inputs are not permitted",
              },
            ],
          },
        },
      }),
    );
    const { result } = renderHook(() => useSetupPreflight(ENTRY));

    // Act
    const outcome = await result.current.runPreflight(VALUES);

    // Assert - dropping the entire requirements envelope would hide a current
    // contract error and incorrectly downgrade the deployment to advisory.
    expect(outcome).toEqual({ status: "unavailable" });
    expect(AutomationService.validateDraft).toHaveBeenCalledTimes(1);
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
    const second = result.current.runPreflight({
      ...VALUES,
      widgetName: "New",
    });
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

  it("marks an in-flight response stale when the setup entry changes", async () => {
    // Arrange
    let resolve!: (value: { valid: true; errors: never[] }) => void;
    vi.mocked(AutomationService.validateDraft).mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    const { result, rerender } = renderHook(
      ({ entry }) => useSetupPreflight(entry),
      { initialProps: { entry: ENTRY } },
    );
    const pending = result.current.runPreflight(VALUES);

    // Act - React Router can reuse the mounted setup route for another id.
    rerender({ entry: createSetupEntry({ id: "different-automation" }) });
    resolve({ valid: true, errors: [] });

    // Assert - the previous automation's verdict cannot validate the new one.
    await expect(pending).resolves.toEqual({ status: "stale" });
  });

  it("marks an in-flight response stale when the same entry id is refreshed", async () => {
    // Arrange
    let resolve!: (value: { valid: true; errors: never[] }) => void;
    vi.mocked(AutomationService.validateDraft).mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    const { result, rerender } = renderHook(
      ({ entry }) => useSetupPreflight(entry),
      { initialProps: { entry: ENTRY } },
    );
    const pending = result.current.runPreflight(VALUES);

    // Act - a catalog refresh can replace an entry without changing its id.
    rerender({
      entry: createSetupEntry({
        description: "Updated catalog description.",
      }),
    });
    resolve({ valid: true, errors: [] });

    // Assert - the old body's verdict cannot validate the refreshed manifest.
    await expect(pending).resolves.toEqual({ status: "stale" });
  });

  it("invalidates a pending draft when a new assisted entry has no draft", async () => {
    // Arrange
    let resolve!: (value: { valid: true; errors: never[] }) => void;
    vi.mocked(AutomationService.validateDraft).mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    const { result, rerender } = renderHook(
      ({ entry }) => useSetupPreflight(entry),
      { initialProps: { entry: ENTRY } },
    );
    const pending = result.current.runPreflight(VALUES);

    // Act - assisted setup has no service draft, but it still supersedes the
    // direct request that was started by the previous entry.
    rerender({
      entry: createSetupEntry({
        setup: {
          ...ENTRY.setup,
          mode: "assisted",
          prompt: undefined,
        },
      }),
    });
    await expect(result.current.runPreflight(VALUES)).resolves.toBeNull();
    resolve({ valid: true, errors: [] });

    // Assert
    await expect(pending).resolves.toEqual({ status: "stale" });
  });

  it("marks an in-flight response stale when the target backend changes", async () => {
    let resolve!: (value: { valid: true; errors: never[] }) => void;
    vi.mocked(AutomationService.validateDraft).mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    const { result, rerender } = renderHook(() => useSetupPreflight(ENTRY));
    const pending = result.current.runPreflight(VALUES);

    activeBackendState.value = {
      backend: {
        id: "cloud-test",
        name: "Cloud test",
        host: "https://app.example.test",
        apiKey: "cloud-key",
        kind: "cloud",
        connectionRevision: 0,
      },
      orgId: "org-2",
    };
    rerender();
    resolve({ valid: true, errors: [] });

    await expect(pending).resolves.toEqual({ status: "stale" });
  });

  it("keeps distinct backend targets separate when their text contains delimiters", async () => {
    let resolve!: (value: { valid: true; errors: never[] }) => void;
    vi.mocked(AutomationService.validateDraft).mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    activeBackendState.value = {
      backend: {
        id: "a",
        name: "First target",
        host: "http://localhost:3000",
        apiKey: "first-key",
        kind: "local",
        connectionRevision: 0,
      },
      orgId: "b:cloud:0:c",
    };
    const { result, rerender } = renderHook(() => useSetupPreflight(ENTRY));
    const pending = result.current.runPreflight(VALUES);

    activeBackendState.value = {
      backend: {
        id: "a:local:0:b",
        name: "Second target",
        host: "https://app.example.test",
        apiKey: "second-key",
        kind: "cloud",
        connectionRevision: 0,
      },
      orgId: "c",
    };
    rerender();
    resolve({ valid: true, errors: [] });

    await expect(pending).resolves.toEqual({ status: "stale" });
  });

  it("marks an in-flight response stale after the setup unmounts", async () => {
    let resolve!: (value: { valid: true; errors: never[] }) => void;
    vi.mocked(AutomationService.validateDraft).mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    const { result, unmount } = renderHook(() => useSetupPreflight(ENTRY));
    const pending = result.current.runPreflight(VALUES);

    unmount();
    resolve({ valid: true, errors: [] });

    await expect(pending).resolves.toEqual({ status: "stale" });
  });
});
