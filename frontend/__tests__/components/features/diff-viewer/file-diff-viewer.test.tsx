import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { FileDiffViewer } from "#/components/features/diff-viewer/file-diff-viewer";

const MOCK_DIFF = { original: "old content", modified: "new content" };
const MOCK_MD_DIFF = {
  original: "# Old Heading",
  modified: "# New Heading\n\nSome **bold** text",
};

let mockDiff = MOCK_DIFF;
let mockIsSuccess = true;
let mockIsLoading = false;

vi.mock("#/hooks/query/use-unified-git-diff", () => ({
  useUnifiedGitDiff: () => ({
    data: mockDiff,
    isLoading: mockIsLoading,
    isSuccess: mockIsSuccess,
    isRefetching: false,
  }),
}));

vi.mock("@monaco-editor/react", () => ({
  DiffEditor: (props: Record<string, unknown>) => (
    <div data-testid="file-diff-viewer" data-original={props.original} data-modified={props.modified} />
  ),
  Editor: (props: Record<string, unknown>) => (
    <div data-testid="file-single-viewer" data-value={props.value} />
  ),
}));

vi.mock("#/components/features/markdown/markdown-renderer", () => ({
  MarkdownRenderer: ({ content }: { content: string }) => (
    <div data-testid="markdown-renderer">{content}</div>
  ),
}));

const expand = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByTestId("collapse"));
};

describe("FileDiffViewer", () => {
  beforeEach(() => {
    mockDiff = MOCK_DIFF;
    mockIsSuccess = true;
    mockIsLoading = false;
  });

  it("starts collapsed with no view mode buttons", () => {
    render(<FileDiffViewer path="src/index.ts" type="M" />);

    expect(screen.queryByTestId("view-mode-old")).not.toBeInTheDocument();
    expect(screen.queryByTestId("view-mode-diff")).not.toBeInTheDocument();
    expect(screen.queryByTestId("view-mode-new")).not.toBeInTheDocument();
  });

  it("shows view mode buttons when expanded", async () => {
    const user = userEvent.setup();
    render(<FileDiffViewer path="src/index.ts" type="M" />);

    await expand(user);

    expect(screen.getByTestId("view-mode-old")).toBeInTheDocument();
    expect(screen.getByTestId("view-mode-diff")).toBeInTheDocument();
    expect(screen.getByTestId("view-mode-new")).toBeInTheDocument();
  });

  it("shows diff editor by default when expanded", async () => {
    const user = userEvent.setup();
    render(<FileDiffViewer path="src/index.ts" type="M" />);

    await expand(user);

    expect(screen.getByTestId("file-diff-viewer")).toBeInTheDocument();
    expect(screen.queryByTestId("file-single-viewer")).not.toBeInTheDocument();
  });

  it("switches to single editor on 'new' mode", async () => {
    const user = userEvent.setup();
    render(<FileDiffViewer path="src/index.ts" type="M" />);

    await expand(user);
    await user.click(screen.getByTestId("view-mode-new"));

    expect(screen.getByTestId("file-single-viewer")).toBeInTheDocument();
    expect(screen.getByTestId("file-single-viewer")).toHaveAttribute("data-value", "new content");
    expect(screen.queryByTestId("file-diff-viewer")).not.toBeInTheDocument();
  });

  it("switches to single editor on 'old' mode", async () => {
    const user = userEvent.setup();
    render(<FileDiffViewer path="src/index.ts" type="M" />);

    await expand(user);
    await user.click(screen.getByTestId("view-mode-old"));

    expect(screen.getByTestId("file-single-viewer")).toBeInTheDocument();
    expect(screen.getByTestId("file-single-viewer")).toHaveAttribute("data-value", "old content");
  });

  it("returns to diff editor when switching back to 'diff' mode", async () => {
    const user = userEvent.setup();
    render(<FileDiffViewer path="src/index.ts" type="M" />);

    await expand(user);
    await user.click(screen.getByTestId("view-mode-new"));
    await user.click(screen.getByTestId("view-mode-diff"));

    expect(screen.getByTestId("file-diff-viewer")).toBeInTheDocument();
    expect(screen.queryByTestId("file-single-viewer")).not.toBeInTheDocument();
  });

  it("renders markdown preview for .md files in 'new' mode", async () => {
    mockDiff = MOCK_MD_DIFF;
    const user = userEvent.setup();
    render(<FileDiffViewer path="README.md" type="M" />);

    await expand(user);
    await user.click(screen.getByTestId("view-mode-new"));

    expect(screen.getByTestId("markdown-preview")).toBeInTheDocument();
    expect(screen.getByTestId("markdown-renderer")).toHaveTextContent(/New Heading/);
    expect(screen.getByTestId("markdown-renderer")).toHaveTextContent(/bold/);
    expect(screen.queryByTestId("file-single-viewer")).not.toBeInTheDocument();
  });

  it("renders markdown preview for .md files in 'old' mode", async () => {
    mockDiff = MOCK_MD_DIFF;
    const user = userEvent.setup();
    render(<FileDiffViewer path="README.md" type="M" />);

    await expand(user);
    await user.click(screen.getByTestId("view-mode-old"));

    expect(screen.getByTestId("markdown-renderer")).toHaveTextContent(MOCK_MD_DIFF.original);
  });

  it("shows diff editor for .md files in 'diff' mode", async () => {
    mockDiff = MOCK_MD_DIFF;
    const user = userEvent.setup();
    render(<FileDiffViewer path="README.md" type="M" />);

    await expand(user);

    expect(screen.getByTestId("file-diff-viewer")).toBeInTheDocument();
    expect(screen.queryByTestId("markdown-preview")).not.toBeInTheDocument();
  });

  it("highlights the active view mode button", async () => {
    const user = userEvent.setup();
    render(<FileDiffViewer path="src/index.ts" type="M" />);

    await expand(user);

    expect(screen.getByTestId("view-mode-diff").className).toContain("bg-neutral-600");
    expect(screen.getByTestId("view-mode-old").className).not.toContain("bg-neutral-600");

    await user.click(screen.getByTestId("view-mode-old"));

    expect(screen.getByTestId("view-mode-old").className).toContain("bg-neutral-600");
    expect(screen.getByTestId("view-mode-diff").className).not.toContain("bg-neutral-600");
  });
});
