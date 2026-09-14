import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { CommitList } from "#/components/features/diff-viewer/commit-list";
import type { GitCommit } from "#/api/open-hands.types";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { count?: number }) => {
      if (
        key === "DIFF_VIEWER$UNCOMMITTED_FILE_COUNT" &&
        typeof options?.count === "number"
      ) {
        return options.count === 1
          ? `${options.count} file`
          : `${options.count} files`;
      }
      return key;
    },
  }),
}));

vi.mock("#/hooks/query/use-commit-changes", () => ({
  useCommitChanges: () => ({
    data: undefined,
    isLoading: false,
    isSuccess: false,
  }),
}));

vi.mock("#/components/features/diff-viewer/diff-change-list", () => ({
  DiffChangeList: ({
    changes,
  }: {
    changes: Array<{ path: string; status: string }>;
  }) => (
    <div data-testid="diff-change-list">
      {changes.map((change) => (
        <div key={change.path}>{change.path}</div>
      ))}
    </div>
  ),
}));

const makeCommit = (overrides: Partial<GitCommit> = {}): GitCommit => ({
  sha: "a".repeat(40),
  shortSha: "aaaaaaa",
  subject: "add logging",
  author: "Agent",
  timestamp: "2026-07-10T12:00:00+07:00",
  ...overrides,
});

describe("CommitList", () => {
  it("renders an Uncommitted accordion row above the commit rows", () => {
    // Arrange / Act
    render(
      <CommitList
        commits={[makeCommit()]}
        hasMore={false}
        uncommittedChanges={[{ path: "src/a.ts", status: "M" }]}
      />,
    );

    // Assert
    expect(screen.getByTestId("uncommitted-changes-row")).toBeInTheDocument();
    expect(screen.getByText("DIFF_VIEWER$UNCOMMITTED")).toBeInTheDocument();
    expect(screen.getByTestId("uncommitted-changes-count")).toHaveTextContent(
      "1 file",
    );
    const rows = screen.getAllByTestId(/^(uncommitted-changes-row|commit-row)$/);
    expect(rows[0]).toHaveAttribute("data-testid", "uncommitted-changes-row");
  });

  it("pluralizes the Uncommitted file count", () => {
    // Arrange / Act
    render(
      <CommitList
        commits={[makeCommit()]}
        hasMore={false}
        uncommittedChanges={[
          { path: "src/a.ts", status: "M" },
          { path: "src/b.ts", status: "A" },
        ]}
      />,
    );

    // Assert
    expect(screen.getByTestId("uncommitted-changes-count")).toHaveTextContent(
      "2 files",
    );
  });

  it("expands Uncommitted into the working-tree file list", async () => {
    // Arrange
    const user = userEvent.setup();
    render(
      <CommitList
        commits={[makeCommit()]}
        hasMore={false}
        uncommittedChanges={[{ path: "src/a.ts", status: "M" }]}
      />,
    );

    // Act
    await user.click(screen.getByTestId("uncommitted-changes-row-toggle"));

    // Assert
    expect(await screen.findByText("src/a.ts")).toBeInTheDocument();
  });

  it("collapses Uncommitted when a commit row is expanded", async () => {
    // Arrange
    const user = userEvent.setup();
    render(
      <CommitList
        commits={[makeCommit()]}
        hasMore={false}
        uncommittedChanges={[{ path: "src/a.ts", status: "M" }]}
      />,
    );
    const uncommittedToggle = screen.getByTestId(
      "uncommitted-changes-row-toggle",
    );
    await user.click(uncommittedToggle);
    expect(uncommittedToggle).toHaveAttribute("aria-expanded", "true");
    expect(await screen.findByText("src/a.ts")).toBeInTheDocument();

    // Act
    await user.click(screen.getByTestId("commit-row-toggle"));

    // Assert — single-open accordion: Uncommitted collapses when a commit opens.
    expect(uncommittedToggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByTestId("commit-row-toggle")).toHaveAttribute(
      "aria-expanded",
      "true",
    );
  });

  it("expands Uncommitted on request and clears the request", () => {
    // Arrange
    const onAutoExpandHandled = vi.fn();

    // Act
    render(
      <CommitList
        commits={[makeCommit()]}
        hasMore={false}
        uncommittedChanges={[{ path: "src/a.ts", status: "M" }]}
        autoExpandUncommitted
        onAutoExpandHandled={onAutoExpandHandled}
      />,
    );

    // Assert
    expect(
      screen.getByTestId("uncommitted-changes-row-toggle"),
    ).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("src/a.ts")).toBeInTheDocument();
    expect(onAutoExpandHandled).toHaveBeenCalled();
  });

  it("still renders Uncommitted when there are no working-tree changes", () => {
    // Arrange / Act
    render(
      <CommitList
        commits={[makeCommit()]}
        hasMore={false}
        uncommittedChanges={[]}
      />,
    );

    // Assert
    expect(screen.getByTestId("uncommitted-changes-row")).toBeInTheDocument();
  });
});
