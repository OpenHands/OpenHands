import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useStartAutomationSetup } from "#/hooks/use-start-automation-setup";

const mocks = vi.hoisted(() => ({
  navigate: vi.fn(),
  mutate: vi.fn(),
  isPending: false,
  getAutomationFormSession: vi.fn(),
  initializeAutomationFormSession: vi.fn(),
  clearAutomationFormSession: vi.fn(),
  markAutomationSetupHandoff: vi.fn(),
}));

vi.mock("#/hooks/mutation/use-create-conversation", () => ({
  useCreateConversation: () => ({
    mutate: mocks.mutate,
    isPending: mocks.isPending,
  }),
}));

vi.mock("#/api/automation-form-session", () => ({
  PENDING_AUTOMATION_SETUP_ID: "pending-new-automation",
  getAutomationFormSession: (...args: unknown[]) =>
    mocks.getAutomationFormSession(...args),
  initializeAutomationFormSession: (...args: unknown[]) =>
    mocks.initializeAutomationFormSession(...args),
  clearAutomationFormSession: (...args: unknown[]) =>
    mocks.clearAutomationFormSession(...args),
}));

vi.mock("#/api/automation-setup-handoff-store", () => ({
  markAutomationSetupHandoff: (...args: unknown[]) =>
    mocks.markAutomationSetupHandoff(...args),
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
    mocks.getAutomationFormSession.mockReset();
    mocks.initializeAutomationFormSession.mockReset();
    mocks.clearAutomationFormSession.mockReset();
    mocks.markAutomationSetupHandoff.mockReset();
    mocks.isPending = false;
  });

  it("creates the conversation from the prompt and keeps the form draft", () => {
    const draft = {
      prompt: "Summarize overnight alerts",
      kind: "prompt" as const,
      form: { name: "Incident digest" },
    };
    mocks.getAutomationFormSession.mockReturnValue(draft);
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
    expect(mocks.initializeAutomationFormSession).toHaveBeenCalledWith(
      "conv-prompt",
      draft,
    );
    expect(mocks.clearAutomationFormSession).toHaveBeenCalledWith(
      "pending-new-automation",
    );
    expect(mocks.markAutomationSetupHandoff).toHaveBeenCalledWith(
      "conv-prompt",
    );
    expect(mocks.navigate).toHaveBeenCalledWith("/conversations/conv-prompt");
  });

  it("sends saved draft tags with the initial prompt", () => {
    const draft = {
      prompt: "Draft prompt",
      kind: "prompt" as const,
      serverDraftId: "draft-1",
      materializedAutomationId: "auto-draft-1",
      form: { name: "Saved draft", prompt: "Draft prompt" },
    };
    mocks.getAutomationFormSession.mockReturnValue(draft);
    mocks.mutate.mockImplementation(async (_payload, options) => {
      await options?.onSuccess?.({ conversation_id: "conv-draft" });
    });
    const { result } = renderHook(() => useStartAutomationSetup());

    result.current.startConversationFromPrompt("  Refine the cadence  ");

    expect(mocks.mutate).toHaveBeenCalledWith(
      {
        query: "Refine the cadence",
        automationSetup: true,
        automationSetupTags: {
          automationsetup: "draft",
          automationdraftid: "draft-1",
          automationmaterializeddraftid: "auto-draft-1",
        },
        entryPoint: "automation_draft_resume",
      },
      expect.any(Object),
    );
    expect(mocks.initializeAutomationFormSession).toHaveBeenCalledWith(
      "conv-draft",
      draft,
    );
    expect(mocks.navigate).toHaveBeenCalledWith("/conversations/conv-draft");
  });

  it("sends edited automation tags with the initial prompt", () => {
    const draft = {
      prompt: "Existing prompt",
      kind: "prompt" as const,
      editingAutomationId: "auto-1",
      form: { name: "Existing automation", prompt: "Existing prompt" },
    };
    mocks.getAutomationFormSession.mockReturnValue(draft);
    mocks.mutate.mockImplementation(async (_payload, options) => {
      await options?.onSuccess?.({ conversation_id: "conv-edit" });
    });
    const { result } = renderHook(() => useStartAutomationSetup());

    result.current.startConversationFromPrompt("  Adjust this automation  ");

    expect(mocks.mutate).toHaveBeenCalledWith(
      {
        query: "Adjust this automation",
        automationSetup: true,
        automationSetupTags: {
          automationsetup: "draft",
          automationeditid: "auto-1",
        },
        entryPoint: "automation_edit",
      },
      expect.any(Object),
    );
    expect(mocks.initializeAutomationFormSession).toHaveBeenCalledWith(
      "conv-edit",
      draft,
    );
    expect(mocks.navigate).toHaveBeenCalledWith("/conversations/conv-edit");
  });

  it("does not create a conversation from an empty prompt", () => {
    const { result } = renderHook(() => useStartAutomationSetup());

    result.current.startConversationFromPrompt("   ");

    expect(mocks.mutate).not.toHaveBeenCalled();
  });
});
