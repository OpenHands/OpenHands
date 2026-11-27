import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { compile } from "@tailwindcss/node";
import { parse } from "postcss";
import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { ChangeAgentContextMenu } from "#/components/features/chat/change-agent-context-menu";

describe("ChangeAgentContextMenu", () => {
  it("marks the active mode and preserves its compact description type", async () => {
    renderWithProviders(
      <ChangeAgentContextMenu
        activeMode="code"
        onClose={vi.fn()}
        onCodeClick={vi.fn()}
        onPlanClick={vi.fn()}
      />,
    );

    const codeOption = screen.getByTestId("code-option");
    const planOption = screen.getByTestId("plan-option");

    expect(codeOption).toHaveClass("bg-interactive-hover");
    expect(planOption).not.toHaveClass("bg-interactive-hover");
    const description = codeOption.querySelector(".whitespace-pre-wrap");
    if (!description) throw new Error("Expected the mode description");
    const compiler = await compile(
      await readFile(resolve("src/tailwind.css"), "utf8"),
      { base: resolve("src"), onDependency: () => {} },
    );
    const rules: string[] = [];
    parse(compiler.build([...description.classList])).walkDecls(
      "font-size",
      (declaration) => {
        const rule = declaration.parent;
        if (rule?.type === "rule" && description.matches(rule.selector)) {
          rules.push(`${rule.selector}{${declaration.toString()}}`);
        }
      },
    );
    const style = document.createElement("style");
    style.textContent = rules.join("\n");
    document.head.appendChild(style);
    try {
      expect(getComputedStyle(description).fontSize).toBe("10px");
    } finally {
      style.remove();
    }

    expect(codeOption.querySelectorAll("svg")).toHaveLength(2);
    expect(planOption.querySelectorAll("svg")).toHaveLength(1);
  });

  it("shows plan as active when conversation mode is plan", () => {
    renderWithProviders(
      <ChangeAgentContextMenu
        activeMode="plan"
        onClose={vi.fn()}
        onCodeClick={vi.fn()}
        onPlanClick={vi.fn()}
      />,
    );

    const codeOption = screen.getByTestId("code-option");
    const planOption = screen.getByTestId("plan-option");

    expect(planOption).toHaveClass("bg-interactive-hover");
    expect(codeOption).not.toHaveClass("bg-interactive-hover");
    expect(planOption.querySelectorAll("svg")).toHaveLength(2);
  });

  it("calls onCodeClick and onClose when code is selected", () => {
    const onClose = vi.fn();
    const onCodeClick = vi.fn();

    renderWithProviders(
      <ChangeAgentContextMenu
        activeMode="plan"
        onClose={onClose}
        onCodeClick={onCodeClick}
        onPlanClick={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByTestId("code-option"));

    expect(onCodeClick).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
