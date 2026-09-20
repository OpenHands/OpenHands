import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  FileDiffViewer,
  MAX_DIFF_EDITOR_HEIGHT_PX,
} from "#/components/features/diff-viewer/file-diff-viewer";

const MOCK_DIFF = { original: "old content", modified: "new content" };
const MOCK_MD_DIFF = {
  original: "# Old Heading",
  modified: "# New Heading\n\nSome **bold** text",
};

let mockDiff = MOCK_DIFF;
let mockIsSuccess = true;
let mockIsLoading = false;
const lifecycleEvents: string[] = [];

function createMockModel(name: string) {
  let disposed = false;

  return {
    isDisposed: () => disposed,
    dispose: vi.fn(() => {
      lifecycleEvents.push(name);
      disposed = true;
    }),
  };
}

function createMockDiffEditor() {
  const original = createMockModel("diff-original-model");
  const modified = createMockModel("diff-modified-model");
  const disposeListeners: (() => void)[] = [];
  const editor = {
    getOriginalEditor: () => ({
      getContentHeight: () => 0,
      onDidContentSizeChange: () => ({ dispose: vi.fn() }),
    }),
    getModifiedEditor: () => ({
      getContentHeight: () => 0,
      onDidContentSizeChange: () => ({ dispose: vi.fn() }),
    }),
    getModel: () => ({ original, modified }),
    onDidDispose: (listener: () => void) => {
      disposeListeners.push(listener);
      return { dispose: vi.fn() };
    },
    dispose: () => {
      lifecycleEvents.push("diff-editor");
      disposeListeners.forEach((listener) => listener());
    },
  };

  return editor;
}

function createMockCodeEditor() {
  const model = createMockModel("single-model");
  const disposeListeners: (() => void)[] = [];
  const editor = {
    getContentHeight: () => 0,
    onDidContentSizeChange: () => ({ dispose: vi.fn() }),
    getModel: () => model,
    onDidDispose: (listener: () => void) => {
      disposeListeners.push(listener);
      return { dispose: vi.fn() };
    },
    dispose: () => {
      lifecycleEvents.push("single-editor");
      disposeListeners.forEach((listener) => listener());
    },
  };

  return editor;
}

vi.mock("#/hooks/query/use-unified-git-diff", () => ({
  useUnifiedGitDiff: () => ({
    data: mockDiff,
    isLoading: mockIsLoading,
    isSuccess: mockIsSuccess,
    isRefetching: false,
  }),
}));

vi.mock("@monaco-editor/react", () => ({
  DiffEditor: (props: Record<string, unknown>) => {
    React.useEffect(() => {
      const editor = createMockDiffEditor();
      const onMount = props.onMount as
        | ((editor: ReturnType<typeof createMockDiffEditor>) => void)
        | undefined;
      onMount?.(editor);
      return () => editor.dispose();
    }, []);

    return (
      <div
        data-testid="file-diff-viewer"
        data-original={props.original}
        data-modified={props.modified}
      />
    );
  },
  Editor: (props: Record<string, unknown>) => {
    React.useEffect(() => {
      const editor = createMockCodeEditor();
      const onMount = props.onMount as
        | ((editor: ReturnType<typeof createMockCodeEditor>) => void)
        | undefined;
      onMount?.(editor);
      return () => editor.dispose();
    }, []);

    return <div data-testid="file-single-viewer" data-value={props.value} />;
  },
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
    lifecycleEvents.length = 0;
  });

  it("disposes diff models after Monaco disposes the diff widget", async () => {
    const user = userEvent.setup();
    const { unmount } = render(<FileDiffViewer path="src/index.ts" type="M" />);

    await expand(user);
    unmount();
    await Promise.resolve();

    expect(lifecycleEvents).toEqual([
      "diff-editor",
      "diff-original-model",
      "diff-modified-model",
    ]);
  });

  it("disposes single-view models after Monaco disposes the editor", async () => {
    const user = userEvent.setup();
    const { unmount } = render(<FileDiffViewer path="src/index.ts" type="M" />);

    await expand(user);
    await user.click(screen.getByTestId("view-mode-new"));
    unmount();
    await Promise.resolve();

    expect(lifecycleEvents).toEqual([
      "diff-editor",
      "diff-original-model",
      "diff-modified-model",
      "single-editor",
      "single-model",
    ]);
  });

  it("caps opened editor panes at 600px", () => {
    expect(MAX_DIFF_EDITOR_HEIGHT_PX).toBe(600);
  });

  it("keeps view mode controls reserved but inert while collapsed", () => {
    render(<FileDiffViewer path="src/index.ts" type="M" />);

    const viewModeGroup = screen.getByTestId("view-mode-diff").parentElement;
    expect(viewModeGroup).toHaveClass("invisible");
    expect(screen.getByTestId("view-mode-old")).toHaveAttribute(
      "tabIndex",
      "-1",
    );
  });

  it("reveals view mode buttons when expanded", async () => {
    const user = userEvent.setup();
    render(<FileDiffViewer path="src/index.ts" type="M" />);

    await expand(user);

    const viewModeGroup = screen.getByTestId("view-mode-diff").parentElement;
    expect(viewModeGroup).not.toHaveClass("invisible");
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

  it("renders a 'file deleted' placeholder when expanded for a deleted file", async () => {
    // Simulate the hook short-circuiting the API call for deleted files.
    mockDiff = undefined as unknown as typeof MOCK_DIFF;
    mockIsSuccess = false;
    const user = userEvent.setup();
    render(<FileDiffViewer path="src/removed.ts" type="D" />);

    await expand(user);

    expect(screen.getByTestId("file-deleted-message")).toBeInTheDocument();
    expect(screen.queryByTestId("file-diff-viewer")).not.toBeInTheDocument();
    expect(screen.queryByTestId("file-single-viewer")).not.toBeInTheDocument();
    // View-mode toolbar should not render for deleted files since there is
    // nothing to switch between.
    expect(screen.queryByTestId("view-mode-old")).not.toBeInTheDocument();
    expect(screen.queryByTestId("view-mode-diff")).not.toBeInTheDocument();
    expect(screen.queryByTestId("view-mode-new")).not.toBeInTheDocument();
  });

  it("renders a deleted file's diff content in commit mode instead of the placeholder", async () => {
    // Arrange — in per-commit mode both sides come from git objects, so a
    // deleted file has real content to show (original vs empty).
    mockDiff = { original: "old content", modified: "" };
    const user = userEvent.setup();
    render(
      <FileDiffViewer path="src/removed.ts" type="D" commit={"a".repeat(40)} />,
    );

    // Act
    await expand(user);

    // Assert
    expect(
      screen.queryByTestId("file-deleted-message"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("file-diff-viewer")).toBeInTheDocument();
  });

  it("reflects the active view mode via aria-pressed on the toggle buttons", async () => {
    const user = userEvent.setup();
    render(<FileDiffViewer path="src/index.ts" type="M" />);

    await expand(user);

    expect(screen.getByTestId("view-mode-diff")).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByTestId("view-mode-old")).toHaveAttribute(
      "aria-pressed",
      "false",
    );

    await user.click(screen.getByTestId("view-mode-old"));

    expect(screen.getByTestId("view-mode-old")).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByTestId("view-mode-diff")).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });
});
