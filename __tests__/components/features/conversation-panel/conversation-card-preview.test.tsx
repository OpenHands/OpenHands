import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { ConversationCardPreview } from "#/components/features/conversation-panel/conversation-card/conversation-card-preview";

const PREVIEW_TITLE = "Conversation 1";

describe("ConversationCardPreview", () => {
  it("always renders each non-redundant tag as a full preview row", () => {
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
        }}
      />,
    );

    expect(screen.queryByTestId("conversation-card-tag-chip")).not.toBeInTheDocument();

    const rows = screen.getAllByTestId("conversation-card-preview-tag-row");
    expect(rows).toHaveLength(5);
    expect(rows[0]).toHaveAttribute("data-tag-key", "git_provider");
    expect(rows[0]).toHaveTextContent("github");
    expect(rows[1]).toHaveAttribute("data-tag-key", "repo_name");
    expect(rows[1]).toHaveTextContent("org/repo");
    expect(rows[2]).toHaveAttribute("data-tag-key", "selected_branch");
    expect(rows[2]).toHaveTextContent("main");
    expect(rows[3]).toHaveAttribute("data-tag-key", "archiveworkspacepath");
    expect(rows[3]).toHaveTextContent("/workspace/project");
    expect(rows[4]).toHaveAttribute("data-tag-key", "owner");
    expect(rows[4]).toHaveTextContent("alice");

    expect(screen.getByText("CONVERSATION_PANEL$PREVIEW_GIT")).toBeInTheDocument();
    expect(screen.getByText("CONVERSATION_PANEL$PREVIEW_REPO")).toBeInTheDocument();
    expect(screen.getByText("CONVERSATION_PANEL$PREVIEW_BRANCH")).toBeInTheDocument();
    expect(
      screen.getByText("CONVERSATION_PANEL$PREVIEW_WORKSPACE"),
    ).toBeInTheDocument();
  });

  it("skips tag rows that duplicate repository / branch / provider fields", () => {
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

    const rows = screen.queryAllByTestId("conversation-card-preview-tag-row");
    expect(rows).toHaveLength(1);
    expect(rows[0]).toHaveAttribute("data-tag-key", "owner");
    expect(rows[0]).toHaveTextContent("alice");
  });

  it("skips archiveworkspacepath when Directory is already shown", () => {
    renderWithProviders(
      <ConversationCardPreview
        title={PREVIEW_TITLE}
        selectedRepository={null}
        workspaceWorkingDir="/workspace/project"
        tags={{ archiveworkspacepath: "/workspace/project", owner: "alice" }}
      />,
    );

    const rows = screen.getAllByTestId("conversation-card-preview-tag-row");
    expect(rows).toHaveLength(1);
    expect(rows[0]).toHaveAttribute("data-tag-key", "owner");
  });

  it("shows full tag values without truncation", () => {
    const longPath =
      "/workspace/project/very/long/nested/directory/path/that-exceeds-chip-budget";
    renderWithProviders(
      <ConversationCardPreview
        title={PREVIEW_TITLE}
        selectedRepository={null}
        tags={{ archiveworkspacepath: longPath }}
      />,
    );

    expect(screen.getByTestId("conversation-card-preview-tag-row")).toHaveTextContent(
      longPath,
    );
  });
});
