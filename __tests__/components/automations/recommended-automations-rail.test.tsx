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
      "github-issue-to-pr",
      "slack-channel-monitor",
      "github-agents-md-maintainer",
      "news-digest",
      "slack-standup-digest",
      "linear-triage-assistant",
      "linear-issue-to-github-pr",
      "linear-issue-to-gitlab-mr",
      "linear-issue-to-bitbucket-pr",
      "jira-issue-to-pr",
      "jira-issue-to-gitlab-mr",
      "research-brief-writer",
      "jira-issue-to-bitbucket-pr",
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
          { name: "GitHub Issue to PR Agent" },
          { name: "Slack channel monitor" },
          { name: "AGENTS.md Maintainer" },
          { name: "Daily news digest" },
          { name: "Slack standup digest" },
          { name: "Linear issue triage assistant" },
          { name: "Jira issue to GitHub PR" },
          { name: "Research brief writer" },
          { name: "Linear issue to GitHub PR" },
          { name: "Linear issue to GitLab MR" },
          { name: "Linear issue to Bitbucket PR" },
          { name: "Jira issue to GitLab MR" },
          { name: "Jira issue to Bitbucket PR" },
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

  describe("drag-to-scroll", () => {
    it("scrolls the rail while dragging with the mouse", () => {
      render(
        <RecommendedAutomationsRail
          installedAutomations={[]}
          onSelect={vi.fn()}
        />,
      );
      const scroller = screen.getByTestId(
        "recommended-automations-rail-scroll",
      );
      mockScrollMetrics(scroller, {
        scrollWidth: 900,
        clientWidth: 320,
        scrollLeft: 100,
      });

      fireEvent.mouseDown(scroller, { button: 0, clientX: 300 });
      fireEvent.mouseMove(document, { clientX: 250, buttons: 1 });

      expect(scroller.scrollLeft).toBe(150);
    });

    it("does not select a card on the click that ends a drag, but allows the next click", () => {
      const onSelect = vi.fn();
      render(
        <RecommendedAutomationsRail
          installedAutomations={[]}
          onSelect={onSelect}
        />,
      );
      const card = screen.getByTestId(
        "recommended-automation-rail-card-slack-standup-digest",
      );

      fireEvent.mouseDown(card, { button: 0, clientX: 300 });
      fireEvent.mouseMove(document, { clientX: 250, buttons: 1 });
      fireEvent.mouseUp(document);
      fireEvent.click(card);

      expect(onSelect).not.toHaveBeenCalled();

      fireEvent.mouseDown(card, { button: 0, clientX: 250 });
      fireEvent.mouseUp(document);
      fireEvent.click(card);

      expect(onSelect).toHaveBeenCalledTimes(1);
    });

    it("treats movement below the drag threshold as a click", () => {
      const onSelect = vi.fn();
      render(
        <RecommendedAutomationsRail
          installedAutomations={[]}
          onSelect={onSelect}
        />,
      );
      const card = screen.getByTestId(
        "recommended-automation-rail-card-slack-standup-digest",
      );

      fireEvent.mouseDown(card, { button: 0, clientX: 300 });
      fireEvent.mouseMove(document, { clientX: 301, buttons: 1 });
      fireEvent.mouseUp(document);
      fireEvent.click(card);

      expect(onSelect).toHaveBeenCalledTimes(1);
    });

    it("ignores drags started with a non-primary mouse button", () => {
      render(
        <RecommendedAutomationsRail
          installedAutomations={[]}
          onSelect={vi.fn()}
        />,
      );
      const scroller = screen.getByTestId(
        "recommended-automations-rail-scroll",
      );
      mockScrollMetrics(scroller, {
        scrollWidth: 900,
        clientWidth: 320,
        scrollLeft: 100,
      });

      fireEvent.mouseDown(scroller, { button: 2, clientX: 300 });
      fireEvent.mouseMove(document, { clientX: 250, buttons: 2 });

      expect(scroller.scrollLeft).toBe(100);
    });

    it("ends the drag once the primary button is no longer held", () => {
      render(
        <RecommendedAutomationsRail
          installedAutomations={[]}
          onSelect={vi.fn()}
        />,
      );
      const scroller = screen.getByTestId(
        "recommended-automations-rail-scroll",
      );
      mockScrollMetrics(scroller, {
        scrollWidth: 900,
        clientWidth: 320,
        scrollLeft: 100,
      });

      fireEvent.mouseDown(scroller, { button: 0, clientX: 300 });
      fireEvent.mouseMove(document, { clientX: 250, buttons: 1 });
      fireEvent.mouseMove(document, { clientX: 200, buttons: 0 });
      fireEvent.mouseMove(document, { clientX: 100, buttons: 1 });

      expect(scroller.scrollLeft).toBe(150);
    });
  });
});
