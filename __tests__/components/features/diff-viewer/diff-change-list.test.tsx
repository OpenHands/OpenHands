import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { DiffChangeList } from "#/components/features/diff-viewer/diff-change-list";

vi.mock("framer-motion", async (importOriginal) => {
  const actual = await importOriginal<typeof import("framer-motion")>();
  return {
    ...actual,
    // Skip exit animations so open/close assertions are synchronous.
    useReducedMotion: () => true,
  };
});

vi.mock("#/hooks/query/use-unified-git-diff", () => ({
  useUnifiedGitDiff: () => ({
    data: { original: "a", modified: "b" },
    isLoading: false,
    isSuccess: true,
    isRefetching: false,
  }),
}));

vi.mock("@monaco-editor/react", () => ({
  DiffEditor: () => <div data-testid="file-diff-viewer" />,
  Editor: () => <div data-testid="file-single-viewer" />,
}));

describe("DiffChangeList", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("keeps only one file expanded at a time", async () => {
    const user = userEvent.setup();
    render(
      <DiffChangeList
        changes={[
          { path: "a.ts", status: "M" },
          { path: "b.ts", status: "A" },
        ]}
      />,
    );

    const [firstToggle, secondToggle] = screen.getAllByTestId("collapse");

    await user.click(firstToggle);
    expect(
      screen.getAllByTestId("file-diff-viewer-outer")[0].querySelector(
        '[data-testid="file-diff-viewer"]',
      ),
    ).toBeTruthy();
    expect(
      screen.getAllByTestId("file-diff-viewer-outer")[1].querySelector(
        '[data-testid="file-diff-viewer"]',
      ),
    ).toBeNull();

    await user.click(secondToggle);
    expect(
      screen.getAllByTestId("file-diff-viewer-outer")[0].querySelector(
        '[data-testid="file-diff-viewer"]',
      ),
    ).toBeNull();
    expect(
      screen.getAllByTestId("file-diff-viewer-outer")[1].querySelector(
        '[data-testid="file-diff-viewer"]',
      ),
    ).toBeTruthy();
  });

  it("collapses the open file when its header is clicked again", async () => {
    const user = userEvent.setup();
    render(
      <DiffChangeList changes={[{ path: "a.ts", status: "M" }]} />,
    );

    await user.click(screen.getByTestId("collapse"));
    expect(screen.getByTestId("file-diff-viewer")).toBeInTheDocument();

    await user.click(screen.getByTestId("collapse"));
    expect(screen.queryByTestId("file-diff-viewer")).not.toBeInTheDocument();
  });
});
