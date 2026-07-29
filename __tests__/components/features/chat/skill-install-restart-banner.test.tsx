import { act, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { SkillInstallRestartBanner } from "#/components/features/chat/skill-install-restart-banner";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useEventStore } from "#/stores/use-event-store";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import type {
  CmdOutputMetadata,
  ExecuteBashObservation,
  ObservationEvent,
} from "#/types/agent-server/core";

// The restart mutation resolves the launch profile through these services
// before creating the conversation; stub them so it deterministically takes
// the legacy agent_settings path (their absence is a supported fallback).
vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  __esModule: true,
  default: { listProfiles: vi.fn().mockRejectedValue(new Error("n/a")) },
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME: "default",
}));
vi.mock("#/api/profiles-service/profiles-service.api", () => ({
  __esModule: true,
  default: {
    listProfiles: vi
      .fn()
      .mockResolvedValue({ profiles: [], active_profile: null }),
  },
}));
vi.mock("#/api/plugins-management-service", () => ({
  __esModule: true,
  default: { listInstalledPlugins: vi.fn().mockResolvedValue([]) },
}));

const CONVERSATION_ID = "test-conversation-id";

const makeInstallEvent = (
  id: string,
  skill = "codereview",
  workspace = "/tmp/demo-ws",
): ObservationEvent<ExecuteBashObservation> => ({
  id,
  timestamp: new Date().toISOString(),
  source: "environment",
  tool_name: "execute_bash",
  tool_call_id: `call-${id}`,
  action_id: `action-${id}`,
  observation: {
    kind: "ExecuteBashObservation",
    content: [
      {
        type: "text",
        text: `✅ Successfully installed '${skill}' to ${workspace}/.agents/skills/${skill}`,
      },
    ],
    command: "python3 fetch_skill.py",
    exit_code: 0,
    error: false,
    timeout: false,
    metadata: {} as CmdOutputMetadata,
  },
});

const addInstallEvent = (id: string) =>
  act(() => {
    useEventStore.getState().addEvent(makeInstallEvent(id));
  });

const renderBanner = () =>
  renderWithProviders(
    <SkillInstallRestartBanner conversationId={CONVERSATION_ID} />,
  );

describe("SkillInstallRestartBanner", () => {
  beforeEach(() => {
    // Load the conversation into the (global) event store so installs are
    // attributed to the rendered conversation.
    useEventStore.getState().clearEventsForConversation(CONVERSATION_ID);
  });

  afterEach(() => {
    __resetActiveStoreForTests();
  });

  it("renders nothing when the conversation has no skill installs", () => {
    renderBanner();

    expect(
      screen.queryByTestId("skill-install-restart-banner"),
    ).not.toBeInTheDocument();
  });

  it("shows the banner after a successful skill install observation", async () => {
    renderBanner();

    addInstallEvent("evt-1");

    expect(
      await screen.findByTestId("skill-install-restart-banner"),
    ).toBeInTheDocument();
  });

  it("hides the banner when dismissed", async () => {
    renderBanner();
    addInstallEvent("evt-1");
    await screen.findByTestId("skill-install-restart-banner");

    await userEvent.click(screen.getByTestId("skill-install-restart-dismiss"));

    expect(
      screen.queryByTestId("skill-install-restart-banner"),
    ).not.toBeInTheDocument();
  });

  it("resurfaces the banner when a new install happens after a dismissal", async () => {
    renderBanner();
    addInstallEvent("evt-1");
    await screen.findByTestId("skill-install-restart-banner");
    await userEvent.click(screen.getByTestId("skill-install-restart-dismiss"));

    addInstallEvent("evt-2");

    expect(
      await screen.findByTestId("skill-install-restart-banner"),
    ).toBeInTheDocument();
  });

  it("renders nothing on a cloud backend", () => {
    setRegisteredBackends([
      {
        id: "cloud-1",
        name: "Cloud",
        host: "https://app.all-hands.dev",
        apiKey: "key",
        kind: "cloud",
      },
    ]);
    setActiveSelection({ backendId: "cloud-1" });
    renderWithProviders(
      <ActiveBackendProvider>
        <SkillInstallRestartBanner conversationId={CONVERSATION_ID} />
      </ActiveBackendProvider>,
    );

    addInstallEvent("evt-1");

    expect(
      screen.queryByTestId("skill-install-restart-banner"),
    ).not.toBeInTheDocument();
  });

  it("starts a new conversation in the install workspace and navigates to it", async () => {
    const createConversationSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue({
        id: "task-1",
        app_conversation_id: "new-conv-1",
        agent_server_url: null,
      } as never);
    const navigate = vi.fn();
    renderWithProviders(
      <SkillInstallRestartBanner conversationId={CONVERSATION_ID} />,
      { navigation: { navigate } },
    );
    addInstallEvent("evt-1");
    await screen.findByTestId("skill-install-restart-banner");

    await userEvent.click(screen.getByTestId("skill-install-restart-action"));

    await waitFor(() => {
      const call = createConversationSpy.mock.lastCall;
      // Positional contract of createConversation: workingDir is #5 and
      // workspaceMode #6 — the parsed install root, reused directly.
      expect(call?.[4]).toBe("/tmp/demo-ws");
      expect(call?.[5]).toBe("local_repo");
      expect(navigate).toHaveBeenCalledWith("/conversations/new-conv-1");
    });
  });
});
