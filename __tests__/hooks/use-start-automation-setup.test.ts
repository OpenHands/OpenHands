import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useStartAutomationSetup } from "#/hooks/use-start-automation-setup";

const mocks = vi.hoisted(() => ({
  navigate: vi.fn(),
  mutate: vi.fn(),
  isPending: false,
  getAutomationSetupDraft: vi.fn(),
  setAutomationSetupDraft: vi.fn(),
  clearAutomationSetupDraft: vi.fn(),
}));

vi.mock("#/hooks/mutation/use-create-conversation", () => ({
  useCreateConversation: () => ({
    mutate: mocks.mutate,
    isPending: mocks.isPending,
  }),
}));

vi.mock("#/api/automation-setup-draft-store", () => ({
  PENDING_AUTOMATION_SETUP_ID: "pending-new-automation",
  getAutomationSetupDraft: (...args: unknown[]) =>
    mocks.getAutomationSetupDraft(...args),
  setAutomationSetupDraft: (...args: unknown[]) =>
    mocks.setAutomationSetupDraft(...args),
  clearAutomationSetupDraft: (...args: unknown[]) =>
    mocks.clearAutomationSetupDraft(...args),
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({ backend: { kind: "local" } }),
}));

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({ navigate: mocks.navigate }),
}));

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({ trackAutomationCreatedButton: vi.fn() }),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

describe("useStartAutomationSetup", () => {
  beforeEach(() => {
    mocks.navigate.mockReset();
    mocks.mutate.mockReset();
    mocks.getAutomationSetupDraft.mockReset();
    mocks.setAutomationSetupDraft.mockReset();
    mocks.clearAutomationSetupDraft.mockReset();
    mocks.isPending = false;
  });

  it("creates the conversation from the prompt and keeps the form draft", () => {
    const draft = {
      prompt: "Summarize overnight alerts",
      kind: "prompt" as const,
      form: { name: "Incident digest" },
    };
    mocks.getAutomationSetupDraft.mockReturnValue(draft);
    mocks.mutate.mockImplementation((_payload, options) => {
      options?.onSuccess?.({ conversation_id: "conv-prompt" });
    });
    const { result } = renderHook(() => useStartAutomationSetup());

    result.current.startConversationFromPrompt("  Watch the inbox  ");

    expect(mocks.mutate).toHaveBeenCalledWith(
      {
        query: "Watch the inbox",
        automationSetup: true,
        entryPoint: "automations_add",
      },
      expect.any(Object),
    );
    expect(mocks.setAutomationSetupDraft).toHaveBeenCalledWith(
      "conv-prompt",
      draft,
    );
    expect(mocks.clearAutomationSetupDraft).toHaveBeenCalledWith(
      "pending-new-automation",
    );
    expect(mocks.navigate).toHaveBeenCalledWith("/conversations/conv-prompt");
  });

  it("does not create a conversation from an empty prompt", () => {
    const { result } = renderHook(() => useStartAutomationSetup());

    result.current.startConversationFromPrompt("   ");

    expect(mocks.mutate).not.toHaveBeenCalled();
  });
});
