import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { SwitchProfileButton } from "#/components/features/chat/switch-profile-button";
import type { LlmProfileSummary } from "#/api/settings-service/profiles-service.api";

// `useTranslation` and `useParams` are mocked globally in vitest.setup.ts.
// We mock the data hooks here to drive the component's branches directly.

const mockUseLlmProfiles = vi.hoisted(() => vi.fn());
const mockUseActiveConversation = vi.hoisted(() => vi.fn());
const mockSwitchAndLog = vi.hoisted(() => vi.fn());

vi.mock("#/hooks/query/use-llm-profiles", () => ({
  useLlmProfiles: () => mockUseLlmProfiles(),
  LLM_PROFILES_QUERY_KEY: "llm-profiles",
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => mockUseActiveConversation(),
}));

vi.mock("#/hooks/mutation/use-switch-llm-profile-and-log", () => ({
  useSwitchLlmProfileAndLog: () => ({
    switchAndLog: mockSwitchAndLog,
    isPending: false,
  }),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: "conv-1" }),
}));

const PROFILES: LlmProfileSummary[] = [
  {
    name: "default",
    model: "anthropic/claude-sonnet-4-6",
    base_url: null,
    api_key_set: true,
  },
  {
    name: "gpt-5",
    model: "openai/gpt-5",
    base_url: null,
    api_key_set: true,
  },
];

const renderButton = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>
        <SwitchProfileButton />
      </QueryClientProvider>
    </MemoryRouter>,
  );
};

const setupHooks = (
  options: {
    profiles?: LlmProfileSummary[];
    activeProfile?: string | null;
    conversationModel?: string | null;
  } = {},
) => {
  mockUseLlmProfiles.mockReturnValue({
    data: {
      profiles: options.profiles ?? PROFILES,
      active_profile: options.activeProfile ?? null,
    },
  });
  mockUseActiveConversation.mockReturnValue({
    data: { llm_model: options.conversationModel ?? null },
  });
};

describe("SwitchProfileButton", () => {
  beforeEach(() => {
    mockSwitchAndLog.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("renders nothing when there are no profiles", () => {
    setupHooks({ profiles: [] });
    renderButton();
    expect(screen.queryByTestId("switch-profile-button")).toBeNull();
  });

  it("shows the matching profile name when conversation.llm_model maps to a profile", () => {
    setupHooks({ conversationModel: "openai/gpt-5" });
    renderButton();
    expect(screen.getByTestId("switch-profile-button")).toHaveTextContent(
      "gpt-5",
    );
  });

  it("falls back to the placeholder when llm_model is set but no profile matches", () => {
    setupHooks({
      conversationModel: "deleted/orphan-model",
      activeProfile: "default",
    });
    renderButton();
    // Should NOT show the user-level default ("default") here — that would
    // misrepresent the running model. Placeholder key is rendered instead.
    const button = screen.getByTestId("switch-profile-button");
    expect(button).toHaveTextContent("LLM$SELECT_MODEL_PLACEHOLDER");
    expect(button).not.toHaveTextContent("default");
  });

  it("uses the user-level active_profile when the conversation has no llm_model yet", () => {
    setupHooks({ conversationModel: null, activeProfile: "default" });
    renderButton();
    expect(screen.getByTestId("switch-profile-button")).toHaveTextContent(
      "default",
    );
  });

  it("toggles the menu open and closed on click", async () => {
    const user = userEvent.setup();
    setupHooks({ conversationModel: "openai/gpt-5" });
    renderButton();
    const button = screen.getByTestId("switch-profile-button");

    expect(button).toHaveAttribute("aria-haspopup", "menu");
    expect(button).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByTestId("switch-profile-context-menu")).toBeNull();

    await user.click(button);
    expect(button).toHaveAttribute("aria-expanded", "true");
    expect(
      screen.getByTestId("switch-profile-context-menu"),
    ).toBeInTheDocument();

    await user.click(button);
    expect(button).toHaveAttribute("aria-expanded", "false");
  });

  it("calls switchAndLog when a non-active profile is selected", async () => {
    const user = userEvent.setup();
    setupHooks({ conversationModel: "openai/gpt-5" }); // gpt-5 is active
    renderButton();
    await user.click(screen.getByTestId("switch-profile-button"));
    await user.click(screen.getByTestId("switch-profile-option-default"));
    expect(mockSwitchAndLog).toHaveBeenCalledWith("conv-1", "default");
  });

  it("does not call switchAndLog when the already-active profile is selected", async () => {
    const user = userEvent.setup();
    setupHooks({ conversationModel: "openai/gpt-5" }); // gpt-5 is active
    renderButton();
    await user.click(screen.getByTestId("switch-profile-button"));
    await user.click(screen.getByTestId("switch-profile-option-gpt-5"));
    expect(mockSwitchAndLog).not.toHaveBeenCalled();
  });
});
