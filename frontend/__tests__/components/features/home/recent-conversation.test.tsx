import { screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { BrowserRouter } from "react-router";
import { RecentConversation } from "#/components/features/home/recent-conversations/recent-conversation";
import type { V1AppConversation } from "#/api/conversation-service/v1-conversation-service.types";
import { V1ExecutionStatus } from "#/types/v1/core";

vi.mock("react-i18next", async () => {
  const actual = await vi.importActual("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => {
        const translations: Record<string, string> = {
          CONVERSATION$AGO: "ago",
          COMMON$NO_REPOSITORY: "No repository",
        };
        return translations[key] || key;
      },
      i18n: {
        changeLanguage: () => new Promise(() => {}),
      },
    }),
  };
});

const baseConversation: V1AppConversation = {
  id: "test-id",
  title: "Test Conversation",
  sandbox_status: "RUNNING",
  execution_status: V1ExecutionStatus.RUNNING,
  updated_at: "2021-10-01T12:00:00Z",
  created_at: "2021-10-01T12:00:00Z",
  selected_repository: null,
  selected_branch: null,
  git_provider: null,
  conversation_url: null,
  created_by_user_id: "user1",
  metrics: null,
  llm_model: null,
  sandbox_id: "sandbox1",
  trigger: null,
  pr_number: [],
  session_api_key: null,
  sub_conversation_ids: [],
};

const renderRecentConversation = (conversation: V1AppConversation) =>
  renderWithProviders(
    <BrowserRouter>
      <RecentConversation conversation={conversation} />
    </BrowserRouter>,
  );

describe("RecentConversation - llm_model", () => {
  it("should render the OpenHands brand label when llm_model is provided", () => {
    renderRecentConversation({
      ...baseConversation,
      llm_model: "anthropic/claude-sonnet-4-20250514",
    });

    const model = screen.getByTestId("recent-conversation-llm-model");
    expect(model).toBeInTheDocument();
    // Visible label is the harness brand; raw model is preserved on hover.
    expect(model).toHaveTextContent("OpenHands");
    expect(model).toHaveAttribute(
      "title",
      "anthropic/claude-sonnet-4-20250514",
    );
    expect(model.querySelector("svg")).toBeInTheDocument();

    const textSpan = model.querySelector("span.truncate");
    expect(textSpan).toBeInTheDocument();
    expect(textSpan).toHaveTextContent("OpenHands");
  });

  it("should render plain 'ACP' for ACP-agent conversations", () => {
    renderRecentConversation({
      ...baseConversation,
      agent_kind: "acp",
      llm_model: null,
    });

    const model = screen.getByTestId("recent-conversation-llm-model");
    expect(model).toHaveTextContent("ACP");
    expect(model).toHaveAttribute("title", "ACP");
  });

  it("should not render the model chip when neither llm_model nor ACP", () => {
    renderRecentConversation(baseConversation);

    expect(
      screen.queryByTestId("recent-conversation-llm-model"),
    ).not.toBeInTheDocument();
  });
});
