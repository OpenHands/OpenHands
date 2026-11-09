import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useStartAutomationSetup } from "#/hooks/use-start-automation-setup";
import { launchConversation } from "#/hooks/mutation/resolve-conversation-launch";
import { useConversationLaunchStore } from "#/stores/conversation-launch-store";
import {
  PENDING_AUTOMATION_SETUP_ID,
  clearAutomationFormSession,
  getAutomationFormSession,
  initializeAutomationFormSession,
} from "#/api/automation-form-session";
import { NavigationProvider } from "#/context/navigation-context";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { AppConversationStartTask } from "#/api/conversation-service/agent-server-conversation-service.types";

vi.mock("#/hooks/mutation/resolve-conversation-launch", () => ({
  launchConversation: vi.fn(),
}));
vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackConversationCreated: vi.fn(),
    trackAutomationCreatedButton: vi.fn(),
  }),
}));

const navigate = vi.fn();
const draft = {
  kind: "prompt" as const,
  prompt: "Review pull requests",
  editingAutomationId: "reviewer-123",
  form: {
    name: "OSS PR Reviewer",
    eventFilter: "contains(repos, repository.full_name)",
  },
};
function mount() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const wrapper = ({ children }: React.PropsWithChildren) =>
    React.createElement(
      QueryClientProvider,
      { client },
      React.createElement(
        ActiveBackendProvider,
        null,
        React.createElement(NavigationProvider, {
          children,
          value: {
            navigate,
            currentPath: "/automations/setup",
            conversationId: null,
            isNavigating: false,
          },
        }),
      ),
    );
  return renderHook(() => useStartAutomationSetup(), { wrapper });
}

beforeEach(() => {
  vi.mocked(launchConversation).mockReset();
  navigate.mockReset();
  useConversationLaunchStore.setState({ launches: {} });
  clearAutomationFormSession(PENDING_AUTOMATION_SETUP_ID);
  clearAutomationFormSession("conv-1");
  setRegisteredBackends([
    {
      id: "local-test",
      kind: "local",
      name: "Local",
      host: "http://localhost:8000",
      apiKey: "test",
    },
  ]);
  setActiveSelection({ backendId: "local-test", orgId: null });
});

describe("automation setup conversation handoff", () => {
  it.each(["local", "cloud"] as const)(
    "opens immediately on %s and carries the selected form through an unmounted launcher",
    async (kind) => {
      setRegisteredBackends([
        {
          id: "local-test",
          kind,
          name: "Test",
          host: "http://localhost:8000",
          apiKey: "test",
        },
      ]);
      initializeAutomationFormSession(PENDING_AUTOMATION_SETUP_ID, draft);
      let finish!: (value: AppConversationStartTask) => void;
      vi.mocked(launchConversation).mockReturnValue(
        new Promise((resolve) => {
          finish = resolve;
        }),
      );
      const hook = mount();
      act(() =>
        hook.result.current.startConversationFromPrompt(
          "Why does this not review docs?",
        ),
      );
      const launch = Object.values(
        useConversationLaunchStore.getState().launches,
      )[0];
      expect(launch.message).toBe("Why does this not review docs?");
      expect(navigate).toHaveBeenCalledWith(`/conversations/${launch.id}`);
      await waitFor(() => expect(launchConversation).toHaveBeenCalled());
      const request = vi.mocked(launchConversation).mock.calls[0][1];
      expect(request.initialUserMsg).toBe("Why does this not review docs?");
      expect(request.automationSetupContext).toContain("reviewer-123");
      expect(request.automationSetupContext).toContain("OSS PR Reviewer");
      expect(request.automationSetupContext).toContain(draft.form.eventFilter);
      expect(request.automationSetupTags).toMatchObject({
        automationeditid: "reviewer-123",
      });
      hook.unmount();
      await act(async () =>
        finish({
          id: "conv-1",
          app_conversation_id: "conv-1",
        } as AppConversationStartTask),
      );
      await waitFor(() =>
        expect(
          useConversationLaunchStore.getState().launches[launch.id].targetId,
        ).toBe("conv-1"),
      );
      expect(getAutomationFormSession("conv-1")).toMatchObject(draft);
      // Completion must not drag the user back after navigating away.
      expect(navigate).toHaveBeenCalledTimes(1);
    },
  );

  it("passes saved draft identity without treating it as an existing automation", async () => {
    initializeAutomationFormSession(PENDING_AUTOMATION_SETUP_ID, {
      prompt: "Review pull requests",
      kind: "prompt",
      serverDraftId: "draft-1",
      materializedAutomationId: "test-automation-1",
    });
    vi.mocked(launchConversation).mockReturnValue(new Promise(() => {}));
    const hook = mount();
    act(() =>
      hook.result.current.startConversationFromPrompt("Finish this draft"),
    );
    await waitFor(() => expect(launchConversation).toHaveBeenCalledOnce());
    expect(vi.mocked(launchConversation).mock.calls[0][1]).toMatchObject({
      automationSetupTags: {
        automationdraftid: "draft-1",
        automationmaterializeddraftid: "test-automation-1",
      },
      automationSetupContext: expect.stringContaining('"draftId":"draft-1"'),
    });
  });

  it("retains the draft and reports failure on the pending conversation", async () => {
    initializeAutomationFormSession(PENDING_AUTOMATION_SETUP_ID, draft);
    vi.mocked(launchConversation).mockRejectedValue(
      new Error("Cannot connect"),
    );
    const hook = mount();
    act(() => hook.result.current.startConversationFromPrompt("Add docs"));
    await waitFor(() =>
      expect(
        Object.values(useConversationLaunchStore.getState().launches)[0]?.error,
      ).toBe("Cannot connect"),
    );
    expect(
      Object.values(useConversationLaunchStore.getState().launches)[0]
        .automationSetupDraft,
    ).toMatchObject(draft);
  });

  it("ignores empty submissions and duplicate sends", async () => {
    vi.mocked(launchConversation).mockReturnValue(new Promise(() => {}));
    const hook = mount();
    act(() => {
      hook.result.current.startConversationFromPrompt(" ");
      hook.result.current.startConversationFromPrompt("Create a digest");
      hook.result.current.startConversationFromPrompt("Create a digest");
    });
    await waitFor(() => expect(launchConversation).toHaveBeenCalledTimes(1));
    expect(vi.mocked(launchConversation).mock.calls[0][1].initialUserMsg).toBe(
      "Create a digest",
    );
    expect(
      Object.keys(useConversationLaunchStore.getState().launches),
    ).toHaveLength(1);
  });
});
