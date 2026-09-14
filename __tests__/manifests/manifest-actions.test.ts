import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useSetupAction } from "#/manifests/manifest-actions";
import type { SetupEntry } from "#/manifests/types";
import { createSetup, createSetupEntry } from "./manifest-test-data";

/**
 * The action bridge for a bundle entry, which is the only path that sends
 * anything before the create call. Packing and the request layer have their own
 * tests; what is exercised here is the order those two are used in.
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

const VALUES = { repository: "OpenHands/automation", widgetName: "Widgets" };

/** The payload the dialog derived for the form, carrying the stand-in path. */
const PAYLOAD = { name: "Widget monitor" };

beforeEach(() => {
  vi.clearAllMocks();
  mocks.packBundle.mockResolvedValue(new Uint8Array([1, 2, 3]));
  vi.mocked(AutomationService.uploadAutomationTarball).mockResolvedValue(
    "oh-internal://uploads/abc",
  );
});

describe("useSetupAction for a bundle entry", () => {
  it("creates against the path the upload returned", async () => {
    // Arrange
    vi.mocked(AutomationService.createAutomationDraft).mockResolvedValue({
      id: "automation-1",
    });
    const { result } = renderHook(() => useSetupAction());

    // Act
    await result.current(ENTRY, VALUES, PAYLOAD);

    // Assert — the stand-in path the form was checked with is replaced by the
    // real one, and the entry decides the endpoint.
    const [body, entry] = vi.mocked(AutomationService.createAutomationDraft)
      .mock.calls[0];
    expect(body.tarball_path).toBe("oh-internal://uploads/abc");
    expect(entry).toBe(ENTRY);
  });

  it("reuses the archive it already uploaded when a create is retried", async () => {
    // Arrange — the service rejects the draft, the user corrects nothing and
    // confirms again. The upload cannot be taken back, so a second one would
    // leave the first behind for good.
    vi.mocked(AutomationService.createAutomationDraft)
      .mockRejectedValueOnce(new Error("Schedule is too frequent"))
      .mockResolvedValueOnce({ id: "automation-1" });
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
    vi.mocked(AutomationService.createAutomationDraft).mockResolvedValue({
      id: "automation-1",
    });
    const { result } = renderHook(() => useSetupAction());

    // Act
    await result.current(ENTRY, VALUES, PAYLOAD);
    await result.current(ENTRY, { ...VALUES, widgetName: "Gadgets" }, PAYLOAD);

    // Assert — the archive carries the answers, so a different answer is a
    // different archive.
    expect(AutomationService.uploadAutomationTarball).toHaveBeenCalledTimes(2);
  });
});
