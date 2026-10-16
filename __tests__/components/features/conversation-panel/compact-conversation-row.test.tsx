import { describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { CompactConversationRow } from "#/components/features/conversation-panel/compact-conversation-row";

vi.mock("react-i18next", async () => {
  const actual = await vi.importActual("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => {
        const translations: Record<string, string> = {
          CONVERSATION_PANEL$TRIGGER_USER: "Started by you",
          CONVERSATION_PANEL$TRIGGER_AUTOMATION: "Started by an automation",
        };
        return translations[key] || key;
      },
      i18n: { changeLanguage: () => new Promise(() => {}) },
    }),
  };
});

const renderRow = (props: Parameters<typeof CompactConversationRow>[0]) =>
  renderWithProviders(<CompactConversationRow {...props} />);

// The compact row reveals its preview through a heroUI tooltip. jsdom does not
// drive heroUI's pointer-based open path, so the preview is opened through the
// trigger's focus, which the tooltip also honours.
const openPreview = async () => {
  const user = userEvent.setup();
  await user.tab();
  await waitFor(() =>
    expect(
      screen.getByTestId("conversation-card-trigger-reason"),
    ).toBeInTheDocument(),
  );
  return user;
};

describe("CompactConversationRow", () => {
  it("shows the localized trigger-reason chip in the hover preview for a recognized trigger", async () => {
    renderRow({
      conversationId: "conversation-1",
      title: "Conversation 1",
      selectedRepository: null,
      lastUpdatedAt: "2021-10-01T12:00:00Z",
      trigger: "automation",
    });

    await openPreview();

    expect(
      screen.getByTestId("conversation-card-trigger-reason"),
    ).toHaveTextContent("Started by an automation");
  });

  it("falls back to the automation chip for a local row with an automation tag", async () => {
    renderRow({
      conversationId: "conversation-1",
      title: "Conversation 1",
      selectedRepository: null,
      lastUpdatedAt: "2021-10-01T12:00:00Z",
      trigger: null,
      tags: { automationname: "Nightly Audit" },
    });

    await openPreview();

    expect(
      screen.getByTestId("conversation-card-trigger-reason"),
    ).toHaveTextContent("Started by an automation");
  });

  it("omits the trigger chip for an unknown trigger with no automation tag", async () => {
    const user = userEvent.setup();
    renderRow({
      conversationId: "conversation-1",
      title: "Conversation 1",
      selectedRepository: null,
      lastUpdatedAt: "2021-10-01T12:00:00Z",
      trigger: "standing_intent" as never,
      tags: { origin: "slack" },
    });

    await user.tab();
    await waitFor(() =>
      expect(screen.getByText("Conversation 1")).toBeInTheDocument(),
    );

    expect(
      screen.queryByTestId("conversation-card-trigger-reason"),
    ).not.toBeInTheDocument();
  });
});
