import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../test-utils";
import { ConversationModelRouterControl } from "#/components/features/conversation/conversation-model-router-control";
import ModelRouterService from "#/api/conversation-service/model-router-service.api";

const activeConversation = vi.hoisted(() => ({
  current: {
    id: "conv-1",
    agent_kind: "openhands" as const,
    llm_model: "default-model",
    conversation_url: "https://runtime.example",
    session_api_key: "session-key",
  },
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: activeConversation.current }),
}));

vi.mock("#/hooks/query/use-llm-profiles", () => ({
  useLlmProfiles: () => ({
    data: {
      profiles: [
        { name: "Reviewer", model: "reviewer-model" },
        { name: "Architect", model: "architect-model" },
      ],
    },
  }),
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "local", kind: "local" },
    orgId: null,
  }),
}));

vi.mock("#/api/conversation-service/model-router-service.api", () => ({
  default: {
    getModelRouter: vi.fn(),
    addProfileRoute: vi.fn(),
    switchRoute: vi.fn(),
    resumeAutoRouting: vi.fn(),
  },
}));

describe("ConversationModelRouterControl", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(ModelRouterService.getModelRouter).mockResolvedValue({
      enabled: false,
      selection: null,
      last_route_key: null,
      routes: [],
    });
  });

  it("adds an LLM profile route to a started conversation", async () => {
    vi.mocked(ModelRouterService.addProfileRoute).mockResolvedValue({
      enabled: true,
      selection: { mode: "auto" },
      last_route_key: null,
      routes: [{ key: "Reviewer", label: "Reviewer", model: "reviewer-model" }],
    });

    renderWithProviders(<ConversationModelRouterControl />);

    await userEvent.click(
      await screen.findByTestId("conversation-model-router-control"),
    );
    await userEvent.click(
      await screen.findByTestId(
        "conversation-model-router-add-profile-Reviewer",
      ),
    );

    await waitFor(() => {
      expect(ModelRouterService.addProfileRoute).toHaveBeenCalledWith(
        "conv-1",
        "https://runtime.example",
        "session-key",
        "Reviewer",
        "Reviewer",
      );
    });
  });

  it("switches to an existing manual route", async () => {
    vi.mocked(ModelRouterService.getModelRouter).mockResolvedValue({
      enabled: true,
      selection: { mode: "auto" },
      last_route_key: null,
      routes: [{ key: "Reviewer", label: "Reviewer", model: "reviewer-model" }],
    });
    vi.mocked(ModelRouterService.switchRoute).mockResolvedValue({
      enabled: true,
      selection: { mode: "manual", manual_route_key: "Reviewer" },
      last_route_key: null,
      routes: [{ key: "Reviewer", label: "Reviewer", model: "reviewer-model" }],
    });

    renderWithProviders(<ConversationModelRouterControl />);

    await userEvent.click(
      await screen.findByTestId("conversation-model-router-control"),
    );
    await userEvent.click(
      await screen.findByTestId("conversation-model-router-route-Reviewer"),
    );

    await waitFor(() => {
      expect(ModelRouterService.switchRoute).toHaveBeenCalledWith(
        "conv-1",
        "https://runtime.example",
        "session-key",
        "Reviewer",
      );
    });
  });
});
