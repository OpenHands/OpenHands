import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { I18nKey } from "#/i18n/declaration";
import { ConversationOverviewAutomationsPanel } from "./conversation-overview-automations-panel";

const mocks = vi.hoisted(() => ({
  launchInChat: vi.fn(),
  closeDrawer: vi.fn(),
  conversation: { id: "conv-1", title: "Fix flaky CI tests" } as {
    id: string;
    title?: string | null;
  },
}));

vi.mock("react-i18next", () => ({
  useTranslation:
    () =>
    (
      key: string,
      options?: Record<string, unknown>,
    ) => {
      if (options && "title" in options) {
        return `${key}(${String(options.title)})`;
      }
      return key;
    },
}));

vi.mock("#/hooks/use-launch-skill-in-chat", () => ({
  useLaunchSkillInChat: () => mocks.launchInChat,
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: mocks.conversation }),
}));

vi.mock("#/hooks/query/use-automation-health", () => ({
  useAutomationHealth: () => ({
    data: { status: "ok" },
    isLoading: false,
    isError: false,
    refetch: vi.fn(),
  }),
}));

vi.mock("#/hooks/query/use-automations", () => ({
  useAutomations: () => ({
    data: { automations: [], total: 0 },
    isLoading: false,
    isError: false,
    refetch: vi.fn(),
  }),
}));

vi.mock("./conversation-overview-drawer-context", () => ({
  useConversationOverviewDrawerOptional: () => ({
    closeDrawer: mocks.closeDrawer,
  }),
}));

vi.mock("#/components/features/automations/empty-state", () => ({
  EmptyState: () => null,
}));

vi.mock("#/components/features/automations/add-automation-modal", () => ({
  AddAutomationModal: () => null,
}));

function renderPanel() {
  return render(<ConversationOverviewAutomationsPanel openAdd={false} />);
}

describe("ConversationOverviewAutomationsPanel", () => {
  it("offers turning the current conversation into an automation", () => {
    renderPanel();

    expect(
      screen.getByTestId("turn-into-automation-action"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("turn-conversation-into-automation-button"),
    ).toHaveTextContent(I18nKey.AUTOMATIONS$TURN_INTO_AUTOMATION);
    expect(
      screen.getByTestId("turn-into-automation-action"),
    ).toHaveTextContent(I18nKey.AUTOMATIONS$TURN_INTO_AUTOMATION_DESC);
  });

  it("seeds the creation flow with the conversation title", () => {
    renderPanel();
    fireEvent.click(
      screen.getByTestId("turn-conversation-into-automation-button"),
    );

    expect(mocks.launchInChat).toHaveBeenCalledTimes(1);
    expect(mocks.launchInChat).toHaveBeenCalledWith(
      `${I18nKey.AUTOMATIONS$CREATE_FROM_CONVERSATION_PROMPT}(Fix flaky CI tests)`,
      expect.any(Function),
    );
  });

  it("falls back to an untitled label when the conversation has no title yet", () => {
    mocks.conversation = { id: "conv-1", title: "" };
    renderPanel();
    fireEvent.click(
      screen.getByTestId("turn-conversation-into-automation-button"),
    );

    expect(mocks.launchInChat).toHaveBeenCalledWith(
      `${I18nKey.AUTOMATIONS$CREATE_FROM_CONVERSATION_PROMPT}(${I18nKey.AUTOMATIONS$CREATE_FROM_CONVERSATION_UNTITLED})`,
      expect.any(Function),
    );
  });

  it("closes the overview drawer once the flow launches", () => {
    renderPanel();
    fireEvent.click(
      screen.getByTestId("turn-conversation-into-automation-button"),
    );

    const onClose = mocks.launchInChat.mock.calls[0][1] as () => void;
    onClose();

    expect(mocks.closeDrawer).toHaveBeenCalledTimes(1);
  });
});
