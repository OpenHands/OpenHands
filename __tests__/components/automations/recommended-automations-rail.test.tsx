import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { I18nKey } from "#/i18n/declaration";
import { RecommendedAutomationsRail } from "#/components/features/automations/recommended-automations-rail";
import { AUTOMATION_STACK_SECTION_BOTTOM_CLASS } from "#/utils/automation-stack-section";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

describe("RecommendedAutomationsRail", () => {
  beforeEach(() => {
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

  it("renders remaining proven workflows before conversation-only extras", () => {
    render(
      <RecommendedAutomationsRail
        installedAutomations={[{ name: "GitHub Code Review Agent" }]}
        onSelect={vi.fn()}
      />,
    );

    const cardIds = screen
      .getAllByTestId(/^recommended-automation-rail-card-/)
      .map((card) =>
        card
          .getAttribute("data-testid")
          ?.replace("recommended-automation-rail-card-", ""),
      );

    expect(cardIds).toEqual([
      "github-repo-monitor",
      "slack-channel-monitor",
      "slack-standup-digest",
      "linear-triage-assistant",
      "jira-issue-to-pr",
      "research-brief-writer",
    ]);
    expect(
      screen.getByText(I18nKey.RECOMMENDED_AUTOMATIONS$SECTION_LABEL),
    ).toBeInTheDocument();
  });

  it("keeps space below the cards when later home sections are empty", () => {
    render(
      <RecommendedAutomationsRail
        installedAutomations={[]}
        onSelect={vi.fn()}
      />,
    );

    expect(screen.getByTestId("recommended-automations-rail")).toHaveClass(
      AUTOMATION_STACK_SECTION_BOTTOM_CLASS,
    );
  });

  it("calls onSelect when a rail card is clicked", async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();

    render(
      <RecommendedAutomationsRail
        installedAutomations={[]}
        onSelect={onSelect}
      />,
    );

    await user.click(
      screen.getByTestId("recommended-automation-rail-card-slack-standup-digest"),
    );

    expect(onSelect).toHaveBeenCalledWith(
      expect.objectContaining({ id: "slack-standup-digest" }),
    );
  });

  it("renders nothing when every recommended automation has been added", () => {
    const { container } = render(
      <RecommendedAutomationsRail
        installedAutomations={[
          { name: "GitHub Code Review Agent" },
          { name: "GitHub repository monitor" },
          { name: "Slack channel monitor" },
          { name: "Slack standup digest" },
          { name: "Linear issue triage assistant" },
          { name: "Jira issue to GitHub PR" },
          { name: "Research brief writer" },
        ]}
        onSelect={vi.fn()}
      />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("keeps a 40px icon row and overlaps multiple logos to the right", () => {
    render(
      <RecommendedAutomationsRail
        installedAutomations={[]}
        onSelect={vi.fn()}
      />,
    );

    const single = screen.getByTestId(
      "recommended-automation-rail-icon-github-pr-reviewer",
    );
    const overlap = screen.getByTestId(
      "recommended-automation-rail-icon-jira-issue-to-pr",
    );

    expect(single).toHaveClass("h-10");
    expect(single).not.toHaveAttribute("data-layout");
    expect(overlap).toHaveClass("h-10", "-space-x-2");
    expect(overlap).toHaveAttribute("data-layout", "overlap");
    expect(overlap).not.toHaveClass("w-10", "bg-surface-raised");
    expect(overlap).not.toHaveAttribute("data-layout", "quadrants");
  });

  describe("clipped-content fades", () => {
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

    it("shows an edge gradient only on the clipped side", () => {
      render(
        <RecommendedAutomationsRail
          installedAutomations={[]}
          onSelect={vi.fn()}
        />,
      );

      const scroller = screen.getByTestId("recommended-automations-rail-scroll");
      const leftFade = screen.getByTestId(
        "recommended-automations-rail-fade-left",
      );
      const rightFade = screen.getByTestId(
        "recommended-automations-rail-fade-right",
      );

      mockScrollMetrics(scroller, {
        scrollWidth: 900,
        clientWidth: 320,
        scrollLeft: 0,
      });
      fireEvent.scroll(scroller);

      expect(rightFade).toHaveAttribute("data-visible", "true");
      expect(leftFade).toHaveAttribute("data-visible", "false");

      mockScrollMetrics(scroller, {
        scrollWidth: 900,
        clientWidth: 320,
        scrollLeft: 580,
      });
      fireEvent.scroll(scroller);

      expect(leftFade).toHaveAttribute("data-visible", "true");
      expect(rightFade).toHaveAttribute("data-visible", "false");
    });
  });
});
