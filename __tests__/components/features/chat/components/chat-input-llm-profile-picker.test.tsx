import { fireEvent, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderWithProviders } from "test-utils";

const useChatInputLlmProfileStateMock = vi.fn();

vi.mock("#/hooks/use-chat-input-llm-profile-state", () => ({
  useChatInputLlmProfileState: () => useChatInputLlmProfileStateMock(),
}));

import { useFreeModelsStore } from "#/stores/free-models-store";
import { I18nKey } from "#/i18n/declaration";
import { ChatInputLlmProfilePicker } from "#/components/features/chat/components/chat-input-llm-profile-picker";

// The test harness renders raw i18n keys and the pill truncates its label, so
// match on the leading slice of the placeholder key rather than its English
// text.
const PLACEHOLDER = I18nKey.LLM$SELECT_MODEL_PLACEHOLDER.slice(0, 18);

const PROFILES = [
  {
    name: "Fast",
    model: "openai/gpt-4o-mini",
    base_url: null,
    api_key_set: true,
  },
  {
    name: "Smart",
    model: "anthropic/claude-opus",
    base_url: null,
    api_key_set: true,
  },
];

const selectProfile = vi.fn();

function state(overrides = {}) {
  return {
    profiles: PROFILES,
    currentProfileName: "Fast",
    currentProfileModel: "openai/gpt-4o-mini",
    isLoading: false,
    isSwitching: false,
    canSwitchProfile: true,
    selectProfile,
    ...overrides,
  };
}

describe("ChatInputLlmProfilePicker", () => {
  beforeEach(() => {
    selectProfile.mockReset();
    useChatInputLlmProfileStateMock.mockReset();
    useChatInputLlmProfileStateMock.mockReturnValue(state());
    useFreeModelsStore.setState({
      freeModels: new Set(),
      defaultModel: null,
      defaultModelReady: false,
    });
  });

  it("renders nothing while loading or when there are no profiles", () => {
    useChatInputLlmProfileStateMock.mockReturnValue(
      state({ profiles: [], isLoading: false }),
    );
    const { container } = renderWithProviders(<ChatInputLlmProfilePicker />);
    expect(container).toBeEmptyDOMElement();
  });

  it("labels the pill with the current profile name", () => {
    renderWithProviders(<ChatInputLlmProfilePicker />);
    expect(screen.getByTestId("chat-input-llm-profile")).toHaveTextContent(
      "Fast",
    );
  });

  it("live-switches to the picked profile", () => {
    renderWithProviders(<ChatInputLlmProfilePicker />);
    fireEvent.click(screen.getByTestId("chat-input-llm-profile"));
    fireEvent.click(screen.getByTestId("chat-input-llm-profile-option-Smart"));

    expect(selectProfile).toHaveBeenCalledWith("Smart");
  });

  it("does not switch when the current profile is re-selected", () => {
    renderWithProviders(<ChatInputLlmProfilePicker />);
    fireEvent.click(screen.getByTestId("chat-input-llm-profile"));
    fireEvent.click(screen.getByTestId("chat-input-llm-profile-option-Fast"));

    expect(selectProfile).not.toHaveBeenCalled();
  });

  it("names the current profile without selectable options when switching is unavailable", () => {
    useChatInputLlmProfileStateMock.mockReturnValue(
      state({ canSwitchProfile: false }),
    );

    renderWithProviders(<ChatInputLlmProfilePicker />);
    fireEvent.click(screen.getByTestId("chat-input-llm-profile"));

    expect(
      screen.getByTestId("chat-input-llm-profile-current"),
    ).toHaveTextContent("Fast");
    expect(
      screen.queryByTestId("chat-input-llm-profile-option-Smart"),
    ).not.toBeInTheDocument();
  });

  it("labels a backend-flagged free OpenHands route in the profile menu", () => {
    useFreeModelsStore.getState().setFlags({
      freeModels: new Set(["openhands/deepseek-v4-flash"]),
      defaultModel: null,
    });
    useChatInputLlmProfileStateMock.mockReturnValue(
      state({
        profiles: [
          {
            name: "Free",
            model: "openhands/deepseek-v4-flash",
            base_url: null,
            api_key_set: true,
          },
        ],
        currentProfileName: "Free",
        currentProfileModel: "openhands/deepseek-v4-flash",
      }),
    );

    renderWithProviders(<ChatInputLlmProfilePicker />);
    fireEvent.click(screen.getByTestId("chat-input-llm-profile"));

    expect(
      screen.getByText("OpenHands DeepSeek V4 Flash (free)"),
    ).toBeInTheDocument();
  });

  it("labels a backend-flagged free OpenHands route in the read-only profile menu", () => {
    useFreeModelsStore.getState().setFlags({
      freeModels: new Set(["openhands/deepseek-v4-flash"]),
      defaultModel: null,
    });
    useChatInputLlmProfileStateMock.mockReturnValue(
      state({
        canSwitchProfile: false,
        currentProfileModel: "openhands/deepseek-v4-flash",
      }),
    );

    renderWithProviders(<ChatInputLlmProfilePicker />);
    fireEvent.click(screen.getByTestId("chat-input-llm-profile"));

    expect(
      screen.getByTestId("chat-input-llm-profile-current"),
    ).toHaveTextContent("OpenHands DeepSeek V4 Flash (free)");
  });

  // A failed or legacy conversation can report an `llm_model` that matches no
  // saved profile; the pill used to collapse to the placeholder even though
  // the conversation plainly has a model (#16263).
  it("names the conversation model when no saved profile matches it", () => {
    useChatInputLlmProfileStateMock.mockReturnValue(
      state({
        currentProfileName: null,
        currentProfileModel: "anthropic/claude-legacy",
      }),
    );

    renderWithProviders(<ChatInputLlmProfilePicker />);

    const pill = screen.getByTestId("chat-input-llm-profile");
    expect(pill).toHaveTextContent("claude-legacy");
    expect(pill).not.toHaveTextContent(PLACEHOLDER);
  });

  it("names the conversation model in the read-only profile menu", () => {
    useChatInputLlmProfileStateMock.mockReturnValue(
      state({
        canSwitchProfile: false,
        currentProfileName: null,
        currentProfileModel: "anthropic/claude-legacy",
      }),
    );

    renderWithProviders(<ChatInputLlmProfilePicker />);
    fireEvent.click(screen.getByTestId("chat-input-llm-profile"));

    expect(
      screen.getByTestId("chat-input-llm-profile-current"),
    ).toHaveTextContent("claude-legacy");
  });

  it("keeps the placeholder when there is neither a profile nor a model", () => {
    useChatInputLlmProfileStateMock.mockReturnValue(
      state({ currentProfileName: null, currentProfileModel: null }),
    );

    renderWithProviders(<ChatInputLlmProfilePicker />);

    expect(screen.getByTestId("chat-input-llm-profile")).toHaveTextContent(
      PLACEHOLDER,
    );
  });

  it("links to the LLM profiles settings page", () => {
    const { container } = renderWithProviders(<ChatInputLlmProfilePicker />);
    fireEvent.click(screen.getByTestId("chat-input-llm-profile"));

    expect(
      container.ownerDocument.querySelector('a[href="/settings/llm"]'),
    ).not.toBeNull();
  });
});
