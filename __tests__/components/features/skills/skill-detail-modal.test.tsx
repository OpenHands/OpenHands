import { render, screen, within, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { SkillDetailModal } from "#/components/features/skills/skill-detail-modal";
import { CustomChatInput } from "#/components/features/chat/custom-chat-input";
import { HOME_PROMPT_DRAFT_KEY } from "#/hooks/chat/use-draft-persistence";
import {
  ADD_SKILL_EXAMPLE_COMMAND,
  ADD_SKILL_SKILL_NAME,
} from "#/constants/skills-docs";
import { useConversationStore } from "#/stores/conversation-store";
import type { SkillInfo } from "#/types/settings";

const navigateMock = vi.fn();

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({
    navigate: navigateMock,
    currentPath: "/skills",
    conversationId: null,
    isNavigating: false,
  }),
  NavigationProvider: ({ children }: { children: React.ReactNode }) => children,
}));

function buildSkill(overrides: Partial<SkillInfo> = {}): SkillInfo {
  return {
    name: "deno",
    type: "knowledge",
    source: "/skills/deno/SKILL.md",
    description: "Deno runtime helper",
    triggers: ["deno"],
    version: "1.0.0",
    license: "MIT",
    compatibility: "Requires Deno 1.40+",
    metadata: { author: "OpenHands" },
    allowed_tools: ["bash"],
    is_agentskills_format: true,
    disable_model_invocation: false,
    content: "# Deno\n\nSkill body.",
    ...overrides,
  };
}

const initialConversationStore = useConversationStore.getState();

describe("SkillDetailModal", () => {
  beforeEach(() => {
    navigateMock.mockReset();
    useConversationStore.setState(initialConversationStore, true);
    sessionStorage.clear();
  });

  it("renders metadata fields and closes on request", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const onToggle = vi.fn();
    const skill = buildSkill();

    render(
      <SkillDetailModal
        skill={skill}
        enabled
        onToggle={onToggle}
        onClose={onClose}
      />,
    );

    const modal = screen.getByTestId("skill-detail-modal");
    expect(
      within(modal).getByTestId(`skill-modal-name-${skill.name}`),
    ).toHaveTextContent(skill.name);
    expect(
      within(modal).getByTestId(`skill-modal-pill-${skill.name}-version`),
    ).toBeInTheDocument();
    expect(
      within(modal).getByTestId("skill-type-badge-knowledge"),
    ).toHaveTextContent("SETTINGS$SKILLS_TYPE_KNOWLEDGE");
    expect(
      within(modal).getByTestId(
        `skill-modal-pill-${skill.name}-metadata-author`,
      ),
    ).toHaveTextContent("OpenHands");
    expect(
      within(modal).getByTestId(`skill-modal-field-content-${skill.name}`),
    ).toHaveValue(skill.content);

    await user.click(within(modal).getByTestId("skill-detail-close"));
    expect(onClose).toHaveBeenCalled();
  });

  it("closes from the top-right close button", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();

    render(
      <SkillDetailModal
        skill={buildSkill()}
        enabled
        onToggle={vi.fn()}
        onClose={onClose}
      />,
    );

    await user.click(screen.getByTestId("skill-detail-modal-close"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("closes the tag popover before closing the detail modal on Escape", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const skill = buildSkill({
      name: "overflow-test",
      triggers: ["one", "two", "three", "four", "five"],
    });

    render(
      <SkillDetailModal
        skill={skill}
        enabled
        onToggle={vi.fn()}
        onClose={onClose}
      />,
    );

    const trigger = screen.getByTestId(
      `skill-modal-pills-${skill.name}-overflow`,
    );
    await user.click(trigger);
    expect(
      screen.getByTestId(`skill-modal-pills-${skill.name}-overflow-popover`),
    ).toBeInTheDocument();

    await user.keyboard("{Escape}");

    expect(
      screen.queryByTestId(`skill-modal-pills-${skill.name}-overflow-popover`),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("skill-detail-modal")).toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();

    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("opens a new chat with the add-skill command from the detail modal", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const setMessageToSend = vi.fn();
    useConversationStore.setState({ setMessageToSend });

    render(
      <SkillDetailModal
        skill={buildSkill({ name: ADD_SKILL_SKILL_NAME })}
        enabled
        onToggle={vi.fn()}
        onClose={onClose}
      />,
    );

    await user.click(
      screen.getByTestId(`skill-detail-use-skill-${ADD_SKILL_SKILL_NAME}`),
    );

    expect(onClose).toHaveBeenCalled();
    expect(navigateMock).toHaveBeenCalledWith("/conversations");
    expect(setMessageToSend).toHaveBeenCalledWith(
      ADD_SKILL_EXAMPLE_COMMAND,
      "home",
    );
  });

  it("pre-fills the Home composer with the skill command, leaving an earlier draft alone until then", async () => {
    // Arrange: a Home draft from earlier in the session, and a message queued
    // for a conversation composer that never consumed it.
    const user = userEvent.setup();
    sessionStorage.setItem(HOME_PROMPT_DRAFT_KEY, "earlier home draft");
    useConversationStore.getState().setMessageToSend("for a conversation");
    const skill = buildSkill({ name: "docker" });

    renderWithProviders(
      <>
        <CustomChatInput onSubmit={vi.fn()} />
        <SkillDetailModal
          skill={skill}
          enabled
          onToggle={vi.fn()}
          onClose={vi.fn()}
        />
      </>,
    );
    const composer = screen.getByTestId("chat-input");
    expect(composer).toHaveTextContent("earlier home draft");

    // Act
    await user.click(
      screen.getByTestId(`skill-detail-use-skill-${skill.name}`),
    );

    // Assert
    await waitFor(() => expect(composer.textContent).toBe("/docker "));
    expect(composer).toHaveFocus();
    expect(useConversationStore.getState().messageToSend).toBeNull();
  });

  it("disables Use skill when the skill is turned off", async () => {
    const user = userEvent.setup();
    const setMessageToSend = vi.fn();
    useConversationStore.setState({ setMessageToSend });
    const skill = buildSkill({ name: ADD_SKILL_SKILL_NAME });

    render(
      <SkillDetailModal
        skill={skill}
        enabled={false}
        onToggle={vi.fn()}
        onClose={vi.fn()}
      />,
    );

    const useSkillButton = screen.getByTestId(
      `skill-detail-use-skill-${skill.name}`,
    );
    expect(useSkillButton).toBeDisabled();

    await user.click(useSkillButton);

    expect(navigateMock).not.toHaveBeenCalled();
    expect(setMessageToSend).not.toHaveBeenCalled();
  });
});
