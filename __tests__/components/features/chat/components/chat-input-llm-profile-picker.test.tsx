import { fireEvent, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderWithProviders } from "test-utils";

const useChatInputLlmProfileStateMock = vi.fn();

vi.mock("#/hooks/use-chat-input-llm-profile-state", () => ({
  useChatInputLlmProfileState: () => useChatInputLlmProfileStateMock(),
}));

// eslint-disable-next-line import/first
import { ChatInputLlmProfilePicker } from "#/components/features/chat/components/chat-input-llm-profile-picker";

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
    selectProfile,
    ...overrides,
  };
}

describe("ChatInputLlmProfilePicker", () => {
  beforeEach(() => {
    selectProfile.mockReset();
    useChatInputLlmProfileStateMock.mockReset();
    useChatInputLlmProfileStateMock.mockReturnValue(state());
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

  it("links to the LLM profiles settings page", () => {
    const { container } = renderWithProviders(<ChatInputLlmProfilePicker />);
    fireEvent.click(screen.getByTestId("chat-input-llm-profile"));

    expect(
      container.ownerDocument.querySelector('a[href="/settings/llm"]'),
    ).not.toBeNull();
  });
});
