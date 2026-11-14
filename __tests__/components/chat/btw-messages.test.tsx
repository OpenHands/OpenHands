import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { BtwMessages } from "#/components/features/chat/btw-messages";
import { I18nKey } from "#/i18n/declaration";
import { useBtwStore } from "#/stores/btw-store";
import { GoalStatusBanner } from "#/components/features/chat/goal-status-banner";
import { useGoalStore } from "#/stores/goal-store";
import { GoalStatus } from "#/types/agent-server/core/events/conversation-state-event";
import { GoalStatusContent } from "#/components/features/chat/goal-status-content";

describe("btw-messages", () => {
  const CONV = "conv-1";
  const entriesFor = (c: string) =>
    useBtwStore.getState().entriesByConversation[c] ?? [];

  describe("<BtwMessages />", () => {
    beforeEach(() => {
      useBtwStore.setState({ entriesByConversation: {} });
    });

    it("renders spinner and no Got it button while pending", () => {
      useBtwStore.getState().addPending(CONV, "why?");
      render(<BtwMessages conversationId={CONV} />);
      expect(screen.getByText("why?")).toBeInTheDocument();
      expect(screen.getByTestId("btw-spinner")).toBeInTheDocument();
      expect(
        screen.queryByRole("button", {
          name: I18nKey.CHAT_INTERFACE$BTW_GOT_IT,
        }),
      ).toBeNull();
    });

    it("renders the response and a Got it button that dismisses on click", async () => {
      const id = useBtwStore.getState().addPending(CONV, "why?");
      useBtwStore.getState().resolve(CONV, id, "because");
      const user = userEvent.setup();
      render(<BtwMessages conversationId={CONV} />);
      expect(screen.getByText(/because/i)).toBeInTheDocument();
      await user.click(
        screen.getByRole("button", {
          name: I18nKey.CHAT_INTERFACE$BTW_GOT_IT,
        }),
      );
      expect(entriesFor(CONV)).toEqual([]);
    });

    it("keeps long responses in a scrollable region", () => {
      const id = useBtwStore.getState().addPending(CONV, "why?");
      const longResponse = Array.from(
        { length: 100 },
        (_, index) => `Response line ${index + 1}`,
      ).join("\n");
      useBtwStore.getState().resolve(CONV, id, longResponse);

      render(<BtwMessages conversationId={CONV} />);

      expect(screen.getByTestId("btw-messages")).toHaveClass(
        "max-h-[40vh]",
        "overflow-y-auto",
        "overscroll-contain",
        "custom-scrollbar-always",
      );
      expect(screen.getByText(/Response line 100/)).toBeInTheDocument();
    });

    it("does not render entries from other conversations", () => {
      useBtwStore.getState().addPending("other-conv", "not mine");
      const { container } = render(<BtwMessages conversationId={CONV} />);
      expect(container).toBeEmptyDOMElement();
    });

    it("renders BTW prefix and pending details through translation keys", async () => {
      useBtwStore.getState().addPending(CONV, "why?");
      const user = userEvent.setup();
      render(<BtwMessages conversationId={CONV} />);
      expect(
        screen.getByText(I18nKey.CHAT_INTERFACE$BTW_PREFIX),
      ).toBeInTheDocument();
      await user.click(screen.getByRole("button", { name: /expand/i }));
      expect(
        screen.getByText(I18nKey.CHAT_INTERFACE$BTW_WAITING_FOR_ANSWER),
      ).toBeInTheDocument();
    });

    it("does not render the pending-state translation key once the entry has resolved", () => {
      const id = useBtwStore.getState().addPending(CONV, "why?");
      useBtwStore.getState().resolve(CONV, id, "because");
      render(<BtwMessages conversationId={CONV} />);
      expect(
        screen.queryByText(I18nKey.CHAT_INTERFACE$BTW_WAITING_FOR_ANSWER),
      ).toBeNull();
    });
  });
});

describe("goal-status-banner", () => {
  const CONV = "conv-1";

  const makeStatus = (overrides: Partial<GoalStatus> = {}): GoalStatus => ({
    active: true,
    status: "running",
    iteration: 1,
    max_iterations: 10,
    objective: "make pytest pass",
    verdict: null,
    ...overrides,
  });

  describe("<GoalStatusBanner />", () => {
    beforeEach(() => {
      useGoalStore.setState({ statusByConversation: {} });
    });

    it("renders nothing when there is no goal status", () => {
      const { container } = render(<GoalStatusBanner conversationId={CONV} />);
      expect(container).toBeEmptyDOMElement();
    });

    it("renders the live banner while the loop is active", () => {
      useGoalStore.getState().setStatus(CONV, makeStatus());
      render(<GoalStatusBanner conversationId={CONV} />);
      expect(screen.getByTestId("goal-status")).toBeInTheDocument();
      expect(screen.getByTestId("goal-spinner")).toBeInTheDocument();
    });

    it("renders nothing once the loop ends (the terminal status renders inline)", () => {
      useGoalStore.getState().setStatus(
        CONV,
        makeStatus({
          active: false,
          status: "complete",
          verdict: { score: 1, complete: true, missing: "" },
        }),
      );
      const { container } = render(<GoalStatusBanner conversationId={CONV} />);
      expect(container).toBeEmptyDOMElement();
    });

    it("does not render a status from another conversation", () => {
      useGoalStore.getState().setStatus("other", makeStatus());
      const { container } = render(<GoalStatusBanner conversationId={CONV} />);
      expect(container).toBeEmptyDOMElement();
    });
  });
});

describe("goal-status-content", () => {
  const makeStatus = (overrides: Partial<GoalStatus> = {}): GoalStatus => ({
    active: true,
    status: "running",
    iteration: 1,
    max_iterations: 10,
    objective: "make pytest pass",
    verdict: null,
    ...overrides,
  });

  describe("<GoalStatusContent />", () => {
    it("shows the objective and a spinner while running", () => {
      render(<GoalStatusContent status={makeStatus()} />);
      expect(screen.getByText("make pytest pass")).toBeInTheDocument();
      expect(screen.getByTestId("goal-spinner")).toBeInTheDocument();
      expect(screen.queryByTestId("goal-done")).toBeNull();
      expect(screen.queryByTestId("goal-ended")).toBeNull();
    });

    it("shows a green check when complete", () => {
      render(
        <GoalStatusContent
          status={makeStatus({
            active: false,
            status: "complete",
            verdict: { score: 1, complete: true, missing: "" },
          })}
        />,
      );
      expect(screen.getByTestId("goal-done")).toBeInTheDocument();
      expect(screen.queryByTestId("goal-spinner")).toBeNull();
      expect(screen.queryByTestId("goal-ended")).toBeNull();
    });

    it("shows a muted cross, not a check, when it ends without completing", () => {
      render(
        <GoalStatusContent
          status={makeStatus({
            active: false,
            status: "capped",
            verdict: {
              score: 0.7,
              complete: false,
              missing: "needs more tests",
            },
          })}
        />,
      );
      expect(screen.getByTestId("goal-ended")).toBeInTheDocument();
      expect(screen.queryByTestId("goal-done")).toBeNull();
    });

    it("expands the judge's missing note for a terminal status", () => {
      render(
        <GoalStatusContent
          status={makeStatus({
            active: false,
            status: "capped",
            verdict: {
              score: 0.7,
              complete: false,
              missing: "needs more tests",
            },
          })}
        />,
      );
      expect(screen.getByText(I18nKey.GOAL$MISSING)).toBeInTheDocument();
    });

    it("keeps the missing note collapsed while running", () => {
      render(
        <GoalStatusContent
          status={makeStatus({
            verdict: { score: 0.5, complete: false, missing: "needs tests" },
          })}
        />,
      );
      expect(screen.queryByText(I18nKey.GOAL$MISSING)).toBeNull();
    });
  });
});
