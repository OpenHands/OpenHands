import { describe, expect, it } from "vitest";
import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";

import { I18nKey } from "#/i18n/declaration";
import { PromptSection } from "#/components/features/automations/detail/prompt-section";

describe("PromptSection", () => {
  it("does not show a toggle for short prompts", () => {
    render(<PromptSection prompt="Short prompt." />);

    expect(screen.getByTestId("automation-prompt-content")).toHaveTextContent(
      "Short prompt.",
    );
    expect(
      screen.queryByTestId("automation-prompt-toggle"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-prompt-fade"),
    ).not.toBeInTheDocument();
  });

  it("collapses long prompts with a fade and expands on toggle", () => {
    const longPrompt = Array.from({ length: 40 }, (_, index) => `Line ${index + 1}.`).join(
      "\n",
    );

    Object.defineProperty(HTMLElement.prototype, "scrollHeight", {
      configurable: true,
      get() {
        return this.textContent?.includes("Line 40.") ? 480 : 120;
      },
    });

    render(<PromptSection prompt={longPrompt} />);

    expect(screen.getByTestId("automation-prompt-fade")).toBeInTheDocument();
    expect(screen.getByTestId("automation-prompt-content")).toHaveClass("max-h-60");
    expect(screen.getByTestId("automation-prompt-toggle")).toHaveTextContent(
      I18nKey.COMMON$VIEW_MORE,
    );

    fireEvent.click(screen.getByTestId("automation-prompt-toggle"));

    expect(
      screen.queryByTestId("automation-prompt-fade"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("automation-prompt-content")).not.toHaveClass(
      "max-h-60",
    );
    expect(screen.getByTestId("automation-prompt-toggle")).toHaveTextContent(
      I18nKey.COMMON$VIEW_LESS,
    );
  });
});
