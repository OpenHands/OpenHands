import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { compile } from "@tailwindcss/node";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";

import { FileTreeView } from "#/components/features/files-tab/file-tree-view";

// FileTreeView composes the recursive TreeNode (extracted into its own file).
// The route-level files-tab test treats the tree as a black box, so these
// cover the tree's own user-facing behavior: empty state, directory
// expand/collapse, and file selection.
describe("FileTreeView", () => {
  it("shows the empty-state message and no tree when there are no files", () => {
    // Arrange + Act
    render(
      <FileTreeView paths={[]} selectedPath={null} onSelectFile={vi.fn()} />,
    );

    // Assert
    expect(screen.getByText("FILES$NO_FILES")).toBeInTheDocument();
    expect(screen.queryByTestId("file-tree-view")).not.toBeInTheDocument();
  });

  it("opens and closes a directory with pointer and keyboard activation", async () => {
    // Arrange
    const user = userEvent.setup();
    render(
      <FileTreeView
        paths={["src/main.ts"]}
        selectedPath={null}
        onSelectFile={vi.fn()}
      />,
    );

    // Assert: the directory row shows but its nested file is hidden.
    expect(screen.getByTestId("file-tree-dir-src")).toBeInTheDocument();
    expect(
      screen.queryByTestId("file-tree-file-src/main.ts"),
    ).not.toBeInTheDocument();

    // Act: expand the directory.
    const directory = screen.getByTestId("file-tree-dir-src");
    await user.click(directory);

    // Assert: the nested file is now visible.
    expect(
      screen.getByTestId("file-tree-file-src/main.ts"),
    ).toBeInTheDocument();
    expect(directory).toHaveAttribute("aria-expanded", "true");

    // The focused native button must also collapse with Space and reopen with Enter.
    await user.keyboard(" ");
    expect(directory).toHaveAttribute("aria-expanded", "false");
    expect(
      screen.queryByTestId("file-tree-file-src/main.ts"),
    ).not.toBeInTheDocument();
    await user.keyboard("{Enter}");
    expect(directory).toHaveAttribute("aria-expanded", "true");
  });

  it("calls onSelectFile with the file path when a file row is activated", async () => {
    // Arrange
    const user = userEvent.setup();
    const onSelectFile = vi.fn();
    render(
      <FileTreeView
        paths={["README.md"]}
        selectedPath={null}
        onSelectFile={onSelectFile}
      />,
    );

    // Act
    const file = screen.getByTestId("file-tree-file-README.md");
    file.focus();
    await user.keyboard("{Enter}");

    // Assert
    expect(onSelectFile).toHaveBeenCalledTimes(1);
    expect(onSelectFile).toHaveBeenCalledWith("README.md");
  });

  it("compiles each rendered row's indentation from its owned depth value", async () => {
    // Arrange: root rows and three nested levels exercise both row types.
    const user = userEvent.setup();
    render(
      <FileTreeView
        paths={["README.md", "src/main.ts", "src/nested/deep/example.ts"]}
        selectedPath="src/nested/deep/example.ts"
        onSelectFile={vi.fn()}
      />,
    );
    for (const path of ["src", "src/nested", "src/nested/deep"]) {
      await user.click(screen.getByTestId(`file-tree-dir-${path}`));
    }
    const rows = [
      [screen.getByTestId("file-tree-dir-src"), 8],
      [screen.getByTestId("file-tree-dir-src/nested"), 20],
      [screen.getByTestId("file-tree-dir-src/nested/deep"), 32],
      [screen.getByTestId("file-tree-file-README.md"), 24],
      [screen.getByTestId("file-tree-file-src/main.ts"), 36],
      [screen.getByTestId("file-tree-file-src/nested/deep/example.ts"), 60],
    ] as const;
    const compiler = await compile(
      await readFile(resolve("src/tailwind.css"), "utf8"),
      { base: resolve("src"), onDependency: () => {} },
    );

    // Act: build only classes emitted by the real recursive tree.
    const css = compiler.build(rows.flatMap(([row]) => [...row.classList]));
    for (const [row, expectedIndent] of rows) {
      const declarations = [
        ...css.matchAll(/\.((?:\\.|[\w-])+)\s*\{([^}]+)\}/g),
      ]
        .filter(([, selector]) =>
          row.classList.contains(selector.replace(/\\(.)/g, "$1")),
        )
        .map(([, , body]) => body)
        .join("\n");
      const padding = /(?:^|\n)\s*padding-left:\s*var\((--[\w-]+)\);/.exec(
        declarations,
      );

      // Assert: the generated property reader consumes this row's real value.
      // A missing utility or a mismatched reader/writer breaks the contract.
      expect(padding).not.toBeNull();
      expect(row.style.getPropertyValue(padding![1])).toBe(
        `${expectedIndent}px`,
      );
    }
  });
});
