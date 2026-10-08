import { act, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DeepPlanPanel } from "#/components/features/chat/deep-plan-panel";
import { I18nKey } from "#/i18n/declaration";
import { useConversationStore } from "#/stores/conversation-store";
import { renderWithProviders } from "../../../../test-utils";

// The panel renders instructions and citation errors through `t()`. Inline the
// English strings for the keys the assertions read so the sentence shape (the
// bracketed citation, the phase name) is exercised rather than a bare key.
const TRANSLATIONS: Record<string, string> = {
  [I18nKey.DEEP_PLAN$TITLE]: "Deep Planning",
  [I18nKey.DEEP_PLAN$EMPTY_MESSAGE]: "Start Deep Planning.",
  [I18nKey.DEEP_PLAN$START]: "Start deep planning",
  [I18nKey.DEEP_PLAN$PHASE_ANALYSIS]: "Analysis",
  [I18nKey.DEEP_PLAN$PHASE_REQUIREMENTS]: "Requirements",
  [I18nKey.DEEP_PLAN$PHASE_DATABASE]: "Database design",
  [I18nKey.DEEP_PLAN$PHASE_BACKEND]: "Backend design",
  [I18nKey.DEEP_PLAN$PHASE_FRONTEND]: "Frontend design",
  [I18nKey.DEEP_PLAN$PHASE_TASKS]: "Tasks",
  [I18nKey.DEEP_PLAN$PHASE_IMPLEMENTATION]: "Implementation",
  [I18nKey.DEEP_PLAN$INSTRUCTION_DATABASE]:
    "PHASE: Database design.\nWrite `database-design.md`.",
  [I18nKey.COMMON$DEEP_PLAN_CONFIRM]: "Confirm phase",
  [I18nKey.COMMON$DEEP_PLAN_UNCOVERED]:
    "Requirements not covered by any task: {{sections}}",
  [I18nKey.DEEP_PLAN$ISSUE_DANGLING]:
    "{{document}} cites [{{ref}}], which no upstream document defines.",
  [I18nKey.DEEP_PLAN$ISSUE_NOT_UPSTREAM]:
    "{{document}} cites [{{ref}}], which is not an upstream document for this phase.",
  [I18nKey.DEEP_PLAN$ISSUE_MISSING_DOCUMENT]:
    "{{document}} cites [{{ref}}], but that document has not been produced yet.",
  [I18nKey.DEEP_PLAN$CONFIRM_BLOCKED]: "Confirm {{phase}} before continuing.",
  [I18nKey.DEEP_PLAN$CONFIRM_MORE_ISSUES]: " (+{{count}} more)",
};

vi.mock("react-i18next", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-i18next")>();
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string, options?: Record<string, unknown>) =>
        (TRANSLATIONS[key] ?? key).replace(/{{(\w+)}}/g, (_, name: string) =>
          String(options?.[name] ?? `{{${name}}}`),
        ),
    }),
  };
});

describe("DeepPlanPanel", () => {
  beforeEach(() => {
    useConversationStore.setState({
      deepPlan: { activePhase: null, confirmed: [], documents: {} },
    });
  });

  it("offers to open the chain when no phase is active", async () => {
    renderWithProviders(<DeepPlanPanel />);

    await userEvent.click(screen.getByRole("button"));

    expect(useConversationStore.getState().deepPlan.activePhase).toBe(
      "analysis",
    );
  });

  it("locks a later phase until every earlier phase is confirmed", () => {
    act(() => useConversationStore.getState().startDeepPlan());

    renderWithProviders(<DeepPlanPanel />);

    expect(screen.getByTestId("deep-plan-phase-analysis")).not.toBeDisabled();
    expect(screen.getByTestId("deep-plan-phase-database")).toBeDisabled();
    expect(screen.getByTestId("deep-plan-phase-database")).toHaveAttribute(
      "data-state",
      "locked",
    );
  });

  it("refuses the checkpoint and names the dangling reference", async () => {
    const store = useConversationStore.getState();
    act(() => {
      store.startDeepPlan();
      store.setDeepPlanDocument("requirements", "## 3.1 Authentication\n");
      store.confirmDeepPlanPhase("analysis");
      store.confirmDeepPlanPhase("requirements");
      store.setDeepPlanDocument("database", "## 2.1 Users [Req 9.9]\n");
    });

    renderWithProviders(<DeepPlanPanel />);

    await userEvent.click(screen.getByTestId("deep-plan-confirm"));

    expect(screen.getByTestId("deep-plan-error")).toHaveTextContent(
      "database-design.md cites [Req 9.9], which no upstream document defines.",
    );
    expect(useConversationStore.getState().deepPlan.activePhase).toBe(
      "database",
    );
  });

  it("clears a stale checkpoint error when the document is re-edited", async () => {
    const store = useConversationStore.getState();
    act(() => {
      store.startDeepPlan();
      store.setDeepPlanDocument("requirements", "## 3.1 Authentication\n");
      store.confirmDeepPlanPhase("analysis");
      store.confirmDeepPlanPhase("requirements");
      store.setDeepPlanDocument("database", "## 2.1 Users [Req 9.9]\n");
    });

    renderWithProviders(<DeepPlanPanel />);

    await userEvent.click(screen.getByTestId("deep-plan-confirm"));
    expect(screen.getByTestId("deep-plan-error")).toBeInTheDocument();

    // Rewriting the database document with a valid citation must drop the
    // error that described the previous (broken) document.
    act(() => {
      useConversationStore
        .getState()
        .setDeepPlanDocument("database", "## 2.1 Users [Req 3.1]\n");
    });

    await waitFor(() =>
      expect(screen.queryByTestId("deep-plan-error")).not.toBeInTheDocument(),
    );
  });

  it("renders the phase instruction through the translation function", () => {
    const store = useConversationStore.getState();
    act(() => {
      store.startDeepPlan();
      store.setDeepPlanDocument("requirements", "## 3.1 Authentication\n");
      store.confirmDeepPlanPhase("analysis");
      store.confirmDeepPlanPhase("requirements");
    });

    renderWithProviders(<DeepPlanPanel />);

    expect(screen.getByText(/PHASE: Database design\./)).toBeInTheDocument();
  });
});
