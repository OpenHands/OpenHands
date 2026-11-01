import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { compile } from "@tailwindcss/node";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect } from "vitest";
import { code as Code } from "#/components/features/markdown/code";

/**
 * jsdom ignores rules inside `@layer` blocks, so lift them out and let the
 * stylesheet's selector specificity decide the cascade.
 */
function unwrapCascadeLayers(css: string): string {
  let out = css.replace(/@layer [\w\s,-]+;/g, "");
  let start = out.search(/@layer [\w-]+\s*\{/);
  while (start !== -1) {
    const open = out.indexOf("{", start);
    let depth = 1;
    let end = open + 1;
    for (; depth > 0; end += 1) {
      if (out[end] === "{") depth += 1;
      if (out[end] === "}") depth -= 1;
    }
    out = out.slice(0, start) + out.slice(open + 1, end - 1) + out.slice(end);
    start = out.search(/@layer [\w-]+\s*\{/);
  }
  return out;
}

describe("code (markdown)", () => {
  it("should render inline code without a copy button", () => {
    render(<Code>inline snippet</Code>);

    expect(screen.getByText("inline snippet")).toBeInTheDocument();
    expect(screen.queryByTestId("copy-to-clipboard")).not.toBeInTheDocument();
  });

  it("should render a multiline code block with a copy button", () => {
    render(<Code>{"line1\nline2"}</Code>);

    expect(screen.getByText("line1 line2")).toBeInTheDocument();
    expect(screen.getByTestId("copy-to-clipboard")).toBeInTheDocument();
  });

  it("should render a syntax-highlighted block with a copy button", () => {
    render(<Code className="language-js">{"console.log('hi')"}</Code>);

    expect(screen.getByTestId("copy-to-clipboard")).toBeInTheDocument();
  });

  it("should copy code block content to clipboard", async () => {
    const user = userEvent.setup();
    render(<Code>{"line1\nline2"}</Code>);

    await user.click(screen.getByTestId("copy-to-clipboard"));

    await waitFor(() =>
      expect(navigator.clipboard.readText()).resolves.toBe("line1\nline2"),
    );
  });

  it("keeps highlighted tokens inline when their Prism type is also a Tailwind utility", async () => {
    // Prism names TOML section headers `token table`, and Tailwind generates
    // `.table { display: table }` whenever the word appears in the sources.
    const compiler = await compile(
      await readFile(resolve("src/tailwind.css"), "utf8"),
      { base: resolve("src"), onDependency: () => {} },
    );
    const style = document.createElement("style");
    style.textContent = unwrapCascadeLayers(
      compiler.build(["table", "static"]),
    );
    document.head.appendChild(style);

    try {
      const { container } = render(
        <Code className="language-toml">
          {'[build-system]\nrequires = ["hatchling"]'}
        </Code>,
      );

      const sectionHeader = container.querySelector(".token.table");
      expect(sectionHeader).toHaveTextContent("build-system");
      expect(getComputedStyle(sectionHeader!).display).toBe("inline");
    } finally {
      style.remove();
    }
  });
});
