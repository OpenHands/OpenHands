import React from "react";
import { readFileSync } from "node:fs";
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { code as Code } from "#/components/features/markdown/code";
import { CodeBlock } from "#/components/features/chat/tool-visualizers/primitives/code-block";
import { HighlightedSourceView } from "#/components/features/files-tab/highlighted-source-view";

describe("Syntax highlighter token layout collisions (#18541)", () => {
  it("defines CSS rules ensuring .token.table, .token.static, and .token.italic override Tailwind collisions", () => {
    const indexCss = readFileSync(`${process.cwd()}/src/index.css`, "utf-8");
    const tailwindCss = readFileSync(
      `${process.cwd()}/src/tailwind.css`,
      "utf-8",
    );
    const combinedCss = `${tailwindCss}\n${indexCss}`;

    // Must explicitly declare .token.table with display: inline to beat Tailwind .table utility
    expect(combinedCss).toMatch(/\.token\.table\s*\{[^}]*display:\s*inline/);
    expect(combinedCss).toMatch(/\.token\.static\s*\{[^}]*position:\s*static/);
    expect(combinedCss).toMatch(/\.token\.italic\s*\{[^}]*font-style:\s*inherit/);
  });

  it("renders TOML section header tokens inline in markdown Code blocks under Tailwind utility styling", () => {
    // Inject Tailwind utility simulation into the test DOM
    const style = document.createElement("style");
    style.setAttribute("data-test-tailwind", "true");
    style.textContent = `
      .table { display: table; }
      .static { position: static; }
      ${readFileSync(`${process.cwd()}/src/index.css`, "utf-8")}
    `;
    document.head.appendChild(style);

    try {
      const { container } = render(
        <Code className="language-toml">{"[build-system]\nrequires = [\"setuptools\"]"}</Code>,
      );

      const tableToken = container.querySelector(".token.table") as HTMLElement;
      expect(tableToken, "Expected a .token.table element in TOML markup").not.toBeNull();
      expect(tableToken.textContent).toContain("build-system");

      const computed = window.getComputedStyle(tableToken);
      expect(computed.display).toBe("inline");
    } finally {
      style.remove();
    }
  });

  it("renders TOML section header tokens inline in HighlightedSourceView", () => {
    const style = document.createElement("style");
    style.setAttribute("data-test-tailwind", "true");
    style.textContent = `
      .table { display: table; }
      ${readFileSync(`${process.cwd()}/src/index.css`, "utf-8")}
    `;
    document.head.appendChild(style);

    try {
      const { container } = render(
        <HighlightedSourceView
          path="pyproject.toml"
          text={"[build-system]\nrequires = [\"setuptools\"]"}
        />,
      );

      const tableToken = container.querySelector(".token.table") as HTMLElement;
      expect(tableToken).not.toBeNull();
      const computed = window.getComputedStyle(tableToken);
      expect(computed.display).toBe("inline");
    } finally {
      style.remove();
    }
  });

  it("renders TOML section header tokens inline in CodeBlock tool primitive", () => {
    const style = document.createElement("style");
    style.setAttribute("data-test-tailwind", "true");
    style.textContent = `
      .table { display: table; }
      ${readFileSync(`${process.cwd()}/src/index.css`, "utf-8")}
    `;
    document.head.appendChild(style);

    try {
      const { container } = render(
        <CodeBlock
          language="toml"
          code={"[build-system]\nrequires = [\"setuptools\"]"}
        />,
      );

      const tableToken = container.querySelector(".token.table") as HTMLElement;
      expect(tableToken).not.toBeNull();
      const computed = window.getComputedStyle(tableToken);
      expect(computed.display).toBe("inline");
    } finally {
      style.remove();
    }
  });
});
