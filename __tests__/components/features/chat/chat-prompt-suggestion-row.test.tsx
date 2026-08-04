import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Sparkles } from "lucide-react";
import { AUTOMATIONS_TEMPLATES_PATH } from "#/components/features/automations/automations-page.constants";
import { ChatPromptSuggestionRow } from "#/components/features/chat/chat-prompt-suggestion-row";
import { NavigationProvider } from "#/context/navigation-context";
import { I18nKey } from "#/i18n/declaration";

function renderRow(ui: React.ReactElement, navigate = vi.fn()) {
  return {
    navigate,
    ...render(
      <NavigationProvider
        value={{
          currentPath: "/",
          conversationId: null,
          isNavigating: false,
          navigate,
        }}
      >
        {ui}
      </NavigationProvider>,
    ),
  };
}

function mockScrollMetrics(
  element: HTMLElement,
  metrics: { scrollWidth: number; clientWidth: number; scrollLeft: number },
) {
  Object.defineProperty(element, "scrollWidth", {
    configurable: true,
    value: metrics.scrollWidth,
  });
  Object.defineProperty(element, "clientWidth", {
    configurable: true,
    value: metrics.clientWidth,
  });
  Object.defineProperty(element, "scrollLeft", {
    configurable: true,
    writable: true,
    value: metrics.scrollLeft,
  });
}

describe("ChatPromptSuggestionRow", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal(
      "ResizeObserver",
      class {
        observe = vi.fn();

        unobserve = vi.fn();

        disconnect = vi.fn();
      },
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders suggestion chips and forwards the prompt on click", async () => {
    const onSuggestionClick = vi.fn();
    const user = userEvent.setup();

    renderRow(
      <ChatPromptSuggestionRow
        onSuggestionClick={onSuggestionClick}
        suggestions={[
          {
            id: "standup-digest",
            labelKey: I18nKey.AUTOMATIONS$STARTER_STANDUP_DIGEST,
            promptKey: I18nKey.AUTOMATIONS$STARTER_STANDUP_DIGEST_PROMPT,
            icon: Sparkles,
          },
        ]}
      />,
    );

    expect(screen.getByTestId("chat-prompt-suggestion-row")).toBeInTheDocument();
    expect(
      screen.getByTestId("chat-prompt-suggestion-standup-digest"),
    ).toHaveTextContent(I18nKey.AUTOMATIONS$STARTER_STANDUP_DIGEST);

    await user.click(screen.getByTestId("chat-prompt-suggestion-standup-digest"));

    expect(onSuggestionClick).toHaveBeenCalledWith(
      I18nKey.AUTOMATIONS$STARTER_STANDUP_DIGEST_PROMPT,
    );
  });

  it("uses rounded-lg chips and animates edge fades while scrolling", () => {
    renderRow(
      <ChatPromptSuggestionRow
        onSuggestionClick={vi.fn()}
        suggestions={[
          {
            id: "standup-digest",
            labelKey: I18nKey.AUTOMATIONS$STARTER_STANDUP_DIGEST,
            promptKey: I18nKey.AUTOMATIONS$STARTER_STANDUP_DIGEST_PROMPT,
            icon: Sparkles,
          },
        ]}
      />,
    );

    const chip = screen.getByTestId("chat-prompt-suggestion-standup-digest");
    expect(chip).toHaveClass("rounded-lg");
    expect(chip).not.toHaveClass("rounded-full");

    const scroller = screen.getByTestId("chat-prompt-suggestion-row");
    const leftFade = screen.getByTestId("chat-prompt-suggestion-fade-left");
    const rightFade = screen.getByTestId("chat-prompt-suggestion-fade-right");

    mockScrollMetrics(scroller, {
      scrollWidth: 900,
      clientWidth: 320,
      scrollLeft: 0,
    });
    fireEvent.scroll(scroller);

    expect(rightFade).toHaveAttribute("data-visible", "true");
    expect(rightFade).toHaveClass("opacity-100");
    expect(leftFade).toHaveAttribute("data-visible", "false");
    expect(leftFade).toHaveClass("opacity-0");

    mockScrollMetrics(scroller, {
      scrollWidth: 900,
      clientWidth: 320,
      scrollLeft: 580,
    });
    fireEvent.scroll(scroller);

    expect(leftFade).toHaveAttribute("data-visible", "true");
    expect(leftFade).toHaveClass("opacity-100");
    expect(rightFade).toHaveAttribute("data-visible", "false");
    expect(rightFade).toHaveClass("opacity-0");
  });

  it("renders a view-all-templates link that navigates to the automations catalog", async () => {
    const navigate = vi.fn();
    const user = userEvent.setup();

    renderRow(
      <ChatPromptSuggestionRow onSuggestionClick={vi.fn()} suggestions={[]} />,
      navigate,
    );

    const viewAllLink = screen.getByTestId(
      "chat-prompt-suggestion-view-all-templates",
    );
    expect(viewAllLink).toHaveAttribute("href", AUTOMATIONS_TEMPLATES_PATH);
    expect(viewAllLink).toHaveTextContent(
      I18nKey.HOME$SUGGESTION_VIEW_ALL_TEMPLATES,
    );

    await user.click(viewAllLink);

    expect(navigate).toHaveBeenCalledWith(AUTOMATIONS_TEMPLATES_PATH, {
      replace: false,
    });
  });
});
