import { screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { BrowserRouter } from "react-router";
import { RecentConversation } from "#/components/features/home/recent-conversations/recent-conversation";
import type { V1AppConversation } from "#/api/conversation-service/v1-conversation-service.types";

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
  execution_status: "RUNNING",
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
};

const renderRecentConversation = (conversation: V1AppConversation) =>
  renderWithProviders(
    <BrowserRouter>
      <RecentConversation conversation={conversation} />
    </BrowserRouter>,
  );

describe("RecentConversation - llm_model", () => {
  it("should render the llm model when provided", () => {
    renderRecentConversation({
      ...baseConversation,
      llm_model: "anthropic/claude-sonnet-4-20250514",
    });

    const model = screen.getByTestId("recent-conversation-llm-model");
    expect(model).toBeInTheDocument();
    expect(model).toHaveTextContent("anthropic/claude-sonnet-4-20250514");
    expect(model).toHaveAttribute(
      "title",
      "anthropic/claude-sonnet-4-20250514",
    );
    expect(model.querySelector("svg")).toBeInTheDocument();

    // Verify truncation structure: text is wrapped in a span with truncate class
    const textSpan = model.querySelector("span.truncate");
    expect(textSpan).toBeInTheDocument();
    expect(textSpan).toHaveTextContent("anthropic/claude-sonnet-4-20250514");
  });

  it("should not render the llm model when not provided", () => {
    renderRecentConversation(baseConversation);

    expect(
      screen.queryByTestId("recent-conversation-llm-model"),
    ).not.toBeInTheDocument();
  });
});
