import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { ConversationCardPreview } from "#/components/features/conversation-panel/conversation-card/conversation-card-preview";

const PREVIEW_TITLE = "Conversation 1";

describe("ConversationCardPreview", () => {
  it("renders free-form tags and skips reserved repo / branch / workspace keys", () => {
    renderWithProviders(
      <ConversationCardPreview
        title={PREVIEW_TITLE}
        selectedRepository={null}
        tags={{
          git_provider: "github",
          repo_name: "org/repo",
          selected_branch: "main",
          archiveworkspacepath: "/workspace/project",
          owner: "alice",
          origin: "slack",
        }}
      />,
    );

    expect(screen.queryByTestId("conversation-card-tag-chip")).not.toBeInTheDocument();

    const rows = screen.getAllByTestId("conversation-card-preview-tag-row");
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveAttribute("data-tag-key", "origin");
    expect(rows[0]).toHaveTextContent("slack");
    expect(rows[1]).toHaveAttribute("data-tag-key", "owner");
    expect(rows[1]).toHaveTextContent("alice");
  });

  it("still shows first-class repo / branch rows from selectedRepository", () => {
    renderWithProviders(
      <ConversationCardPreview
        title={PREVIEW_TITLE}
        selectedRepository={{
          selected_repository: "org/repo",
          selected_branch: "main",
          git_provider: "github",
        }}
        tags={{
          git_provider: "github",
          repo_name: "org/repo",
          selected_branch: "main",
          owner: "alice",
        }}
      />,
    );

    expect(screen.getByText("org/repo")).toBeInTheDocument();
    expect(screen.getByText("main")).toBeInTheDocument();
    const rows = screen.getAllByTestId("conversation-card-preview-tag-row");
    expect(rows).toHaveLength(1);
    expect(rows[0]).toHaveAttribute("data-tag-key", "owner");
  });

  it("shows Directory from workspaceWorkingDir instead of a workspace tag", () => {
    renderWithProviders(
      <ConversationCardPreview
        title={PREVIEW_TITLE}
        selectedRepository={null}
        workspaceWorkingDir="/workspace/project"
        tags={{ archiveworkspacepath: "/workspace/project", owner: "alice" }}
      />,
    );

    expect(screen.getByText("/workspace/project")).toBeInTheDocument();
    const rows = screen.getAllByTestId("conversation-card-preview-tag-row");
    expect(rows).toHaveLength(1);
    expect(rows[0]).toHaveAttribute("data-tag-key", "owner");
  });

  it("shows full tag values without truncation", () => {
    const longValue = "a-very-long-custom-tag-value-that-exceeds-chip-budget";
    renderWithProviders(
      <ConversationCardPreview
        title={PREVIEW_TITLE}
        selectedRepository={null}
        tags={{ custom_token: longValue }}
      />,
    );

    expect(screen.getByTestId("conversation-card-preview-tag-row")).toHaveTextContent(
      longValue,
    );
  });

  it("styles the model row with the OpenHands brand icon", () => {
    renderWithProviders(
      <ConversationCardPreview
        title={PREVIEW_TITLE}
        selectedRepository={null}
        llmModel="openhands/claude-opus-4-5-20251101"
      />,
    );

    expect(
      screen.getByTestId("conversation-card-preview-model"),
    ).toHaveTextContent("openhands/claude-opus-4-5-20251101");
    expect(
      screen.getByTestId("agent-brand-icon-openhands"),
    ).toBeInTheDocument();
  });

  it("uses the ACP provider mark on the model row for ACP conversations", () => {
    // The card chip resolves the ACP brand mark; the hovercard for the same
    // conversation must not contradict it with the OpenHands wordmark.
    renderWithProviders(
      <ConversationCardPreview
        title={PREVIEW_TITLE}
        selectedRepository={null}
        llmModel="claude-opus-4-5"
        agentKind="acp"
        acpServer="claude-code"
      />,
    );

    expect(
      screen.getByTestId("agent-brand-icon-claude-code"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("agent-brand-icon-openhands"),
    ).not.toBeInTheDocument();
  });

  it("shows bare tags (empty value) with an em dash in the hovercard", () => {
    renderWithProviders(
      <ConversationCardPreview
        title={PREVIEW_TITLE}
        selectedRepository={null}
        tags={{ appmode: "work", worktools: "", workwsid: "abc" }}
      />,
    );

    const rows = screen.getAllByTestId("conversation-card-preview-tag-row");
    expect(rows).toHaveLength(3);
    expect(rows[0]).toHaveAttribute("data-tag-key", "appmode");
    expect(rows[0]).toHaveTextContent("work");
    expect(rows[1]).toHaveAttribute("data-tag-key", "worktools");
    expect(rows[1]).toHaveTextContent("—");
    expect(rows[2]).toHaveAttribute("data-tag-key", "workwsid");
    expect(rows[2]).toHaveTextContent("abc");
  });
});
