import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useSetupAction } from "#/manifests/manifest-actions";
import type { SetupEntry, SetupRequestBody } from "#/manifests/types";
import { createSetup, createSetupEntry } from "./manifest-test-data";

/**
 * The action bridge for a direct setup. Packing and the request layer have
 * their own tests; what is exercised here is the safe create/park order and
 * the bundle upload that can precede it.
 */
const mocks = vi.hoisted(() => ({
  packBundle: vi.fn(),
}));

vi.mock("#/manifests/manifest-bundle", () => ({
  packBundle: mocks.packBundle,
}));

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    uploadAutomationTarball: vi.fn(),
    createAutomationDraft: vi.fn(),
    updateAutomation: vi.fn(),
    deleteAutomation: vi.fn(),
  },
}));

vi.mock("#/hooks/mutation/use-create-conversation", () => ({
  useCreateConversation: () => ({ mutateAsync: vi.fn() }),
}));

vi.mock("#/stores/conversation-store", () => ({
  useConversationStore: (select: (state: unknown) => unknown) =>
    select({ setMessageToSend: vi.fn() }),
}));

const ENTRY: SetupEntry = createSetupEntry({
  setup: createSetup({
    prompt: undefined,
    bundle: {
      version: "1.0.0",
      entrypoint: "python3 main.py",
      files: { "main.py": "skills/widget-monitor/scripts/main.py" },
      config: { repos: ["{{form.repository}}"] },
    },
  }),
});

const PROMPT_ENTRY: SetupEntry = createSetupEntry();

const VALUES = {
  schedule: "*/15 * * * *",
  repository: "OpenHands/automation",
  widgetName: "Widgets",
};

/** The payload the dialog derived for the form, carrying the stand-in path. */
const PAYLOAD = { name: "Widget monitor" };

function echoNewAutomation(body: SetupRequestBody) {
  return Promise.resolve({
    id: "automation-1",
    trigger: body.trigger,
    enabled: body.enabled ?? true,
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  mocks.packBundle.mockResolvedValue(new Uint8Array([1, 2, 3]));
  vi.mocked(AutomationService.uploadAutomationTarball).mockResolvedValue(
    "oh-internal://uploads/abc",
  );
  vi.mocked(AutomationService.createAutomationDraft).mockImplementation(
    echoNewAutomation,
  );
  vi.mocked(AutomationService.updateAutomation).mockImplementation(
    async (id, body) =>
      ({
        id,
        trigger: body.trigger,
        enabled: body.enabled,
      }) as never,
  );
  vi.mocked(AutomationService.deleteAutomation).mockResolvedValue(undefined);
});

describe("useSetupAction for a direct entry", () => {
  it("parks a bundle behind a unique event before applying its real trigger disabled", async () => {
    // Arrange
    const { result } = renderHook(() => useSetupAction());

    // Act
    const action = await result.current(ENTRY, VALUES, PAYLOAD);

    // Assert — raw bundle creation cannot accept `enabled`, so the create gets
    // an inert unique trigger and only the follow-up PATCH installs the real
    // trigger and disabled state together.
    const [createBody, entry] = vi.mocked(
      AutomationService.createAutomationDraft,
    ).mock.calls[0];
    expect(createBody.tarball_path).toBe("oh-internal://uploads/abc");
    expect(createBody).not.toHaveProperty("enabled");
    expect(createBody.trigger).toMatchObject({
      type: "event",
      on: expect.stringMatching(/^pending\./),
    });
    expect(entry).toBe(ENTRY);
    expect(AutomationService.updateAutomation).toHaveBeenCalledWith(
      "automation-1",
      expect.objectContaining({
        trigger: { type: "cron", schedule: "*/15 * * * *" },
        enabled: false,
      }),
    );
    expect(action.created).toBe(true);
    expect(action.response).toMatchObject({
      id: "automation-1",
      enabled: false,
    });
  });

  it("also asks preset creation to start disabled before parking the real trigger", async () => {
    // Arrange
    const { result } = renderHook(() => useSetupAction());
    const payload: SetupRequestBody = {
      name: "Widget monitor",
      prompt: "Report on widgets",
      trigger: { type: "cron", schedule: "*/15 * * * *" },
    };

    // Act
    await result.current(PROMPT_ENTRY, VALUES, payload);

    // Assert — preset endpoints support `enabled=false` atomically, while the
    // unique pending trigger still proves whether this request created it.
    const [createBody] = vi.mocked(AutomationService.createAutomationDraft).mock
      .calls[0];
    expect(createBody.enabled).toBe(false);
    expect(createBody.trigger).toMatchObject({
      type: "event",
      on: expect.stringMatching(/^pending\./),
    });
    expect(AutomationService.updateAutomation).toHaveBeenCalledWith(
      "automation-1",
      {
        trigger: { type: "cron", schedule: "*/15 * * * *" },
        enabled: false,
      },
    );
  });

  it("leaves an idempotently returned existing automation untouched", async () => {
    // Arrange — the service returns the already-existing template instead of
    // echoing this request's unique pending trigger.
    vi.mocked(AutomationService.createAutomationDraft).mockResolvedValue({
      id: "existing-1",
      trigger: { type: "cron", schedule: "0 9 * * *" },
      enabled: true,
    });
    const { result } = renderHook(() => useSetupAction());

    // Act
    const action = await result.current(PROMPT_ENTRY, VALUES, {
      name: "Widget monitor",
      prompt: "Report on widgets",
      trigger: { type: "cron", schedule: "*/15 * * * *" },
    });

    // Assert — an existing live automation must never be disabled or rewritten
    // merely because its setup template was opened again.
    expect(action.created).toBe(false);
    expect(action.response).toMatchObject({ id: "existing-1", enabled: true });
    expect(AutomationService.updateAutomation).not.toHaveBeenCalled();
    expect(AutomationService.deleteAutomation).not.toHaveBeenCalled();
  });

  it("cleans up a newly-created parked record if finalizing its disabled state fails", async () => {
    // Arrange
    vi.mocked(AutomationService.updateAutomation).mockRejectedValue(
      new Error("Patch failed"),
    );
    const { result } = renderHook(() => useSetupAction());

    // Act / Assert
    await expect(
      result.current(PROMPT_ENTRY, VALUES, {
        name: "Widget monitor",
        prompt: "Report on widgets",
        trigger: { type: "cron", schedule: "*/15 * * * *" },
      }),
    ).rejects.toThrow("Patch failed");
    expect(AutomationService.deleteAutomation).toHaveBeenCalledWith(
      "automation-1",
    );
  });

  it("reuses the archive it already uploaded when a create is retried", async () => {
    // Arrange — the service rejects the draft, the user corrects nothing and
    // confirms again. The upload cannot be taken back, so a second one would
    // leave the first behind for good.
    vi.mocked(AutomationService.createAutomationDraft)
      .mockRejectedValueOnce(new Error("Schedule is too frequent"))
      .mockImplementationOnce(echoNewAutomation);
    const { result } = renderHook(() => useSetupAction());

    // Act
    await expect(result.current(ENTRY, VALUES, PAYLOAD)).rejects.toThrow();
    await result.current(ENTRY, VALUES, PAYLOAD);

    // Assert
    expect(AutomationService.uploadAutomationTarball).toHaveBeenCalledTimes(1);
    expect(AutomationService.createAutomationDraft).toHaveBeenCalledTimes(2);
  });

  it("packs and uploads again once an answer changes", async () => {
    // Arrange
    const { result } = renderHook(() => useSetupAction());

    // Act
    await result.current(ENTRY, VALUES, PAYLOAD);
    await result.current(ENTRY, { ...VALUES, widgetName: "Gadgets" }, PAYLOAD);

    // Assert — the archive carries the answers, so a different answer is a
    // different archive.
    expect(AutomationService.uploadAutomationTarball).toHaveBeenCalledTimes(2);
  });
});
