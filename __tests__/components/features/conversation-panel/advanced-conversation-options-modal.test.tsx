import { I18nKey } from "#/i18n/declaration";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, screen, within } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { AdvancedConversationOptionsModal } from "#/components/features/conversation-panel/advanced-conversation-options-modal";
import { useConversationPanelPreferencesStore } from "#/stores/conversation-panel-preferences-store";
import React from "react";
import { BudgetDisplay } from "#/components/features/conversation-panel/budget-display";
import i18next from "i18next";
import { translationResources } from "#/i18n/resources";
import { ConfirmStopModal } from "#/components/features/conversation-panel/confirm-stop-modal";
import userEvent from "@testing-library/user-event";
import { ConversationActiveTagFilters } from "#/components/features/conversation-panel/conversation-active-tag-filters";
import { UNNAMED_AUTOMATION_FACET } from "#/components/features/conversation-panel/conversation-panel-list-helpers";

describe("advanced-conversation-options-modal", () => {
  function mockVerticalScrollMetrics(
    element: HTMLElement,
    metrics: { scrollHeight: number; clientHeight: number; scrollTop: number },
  ) {
    Object.defineProperty(element, "scrollHeight", {
      configurable: true,
      value: metrics.scrollHeight,
    });
    Object.defineProperty(element, "clientHeight", {
      configurable: true,
      value: metrics.clientHeight,
    });
    Object.defineProperty(element, "scrollTop", {
      configurable: true,
      writable: true,
      value: metrics.scrollTop,
    });
  }

  describe("AdvancedConversationOptionsModal", () => {
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
      useConversationPanelPreferencesStore.setState({
        showOlderConversations: true,
        olderConversationCutoff: "7d",
      });
      vi.unstubAllGlobals();
    });

    it("uses the standard modal width", () => {
      renderWithProviders(
        <AdvancedConversationOptionsModal
          open
          onClose={vi.fn()}
          backendKind="local"
          automationNameFacets={[]}
        />,
      );

      expect(
        screen.getByTestId("advanced-conversation-options-modal"),
      ).toHaveClass("w-130");
    });

    it("keeps thread scope in its own section, apart from inclusion toggles", () => {
      renderWithProviders(
        <AdvancedConversationOptionsModal
          open
          onClose={vi.fn()}
          backendKind="local"
          automationNameFacets={[]}
        />,
      );

      expect(
        screen.getByText("CONVERSATION_PANEL$THREADS"),
      ).toBeInTheDocument();
      expect(screen.getByTestId("scope-all")).toHaveAttribute(
        "role",
        "menuitemradio",
      );
      expect(screen.getByTestId("scope-relevant")).toHaveAttribute(
        "role",
        "menuitemradio",
      );
      expect(screen.getByTestId("toggle-show-archived")).toHaveAttribute(
        "role",
        "menuitemcheckbox",
      );
      expect(screen.getByTestId("toggle-older-conversations")).toHaveAttribute(
        "role",
        "menuitemcheckbox",
      );
      expect(screen.getByTestId("toggle-older-conversations")).toHaveAttribute(
        "aria-checked",
        "false",
      );
    });

    it("hides conversations older than the selected interval", () => {
      renderWithProviders(
        <AdvancedConversationOptionsModal
          open
          onClose={vi.fn()}
          backendKind="local"
          automationNameFacets={[]}
        />,
      );

      fireEvent.click(screen.getByTestId("toggle-older-conversations"));
      expect(
        useConversationPanelPreferencesStore.getState().showOlderConversations,
      ).toBe(false);

      fireEvent.click(
        within(screen.getByTestId("older-conversation-cutoff")).getByTestId(
          "dropdown-trigger",
        ),
      );
      fireEvent.click(screen.getByTestId("older-conversation-cutoff-1d"));
      expect(
        useConversationPanelPreferencesStore.getState().olderConversationCutoff,
      ).toBe("1d");
    });

    it("closes from the top-right X", () => {
      const onClose = vi.fn();
      renderWithProviders(
        <AdvancedConversationOptionsModal
          open
          onClose={onClose}
          backendKind="local"
          automationNameFacets={[]}
        />,
      );

      fireEvent.click(screen.getByTestId("advanced-options-modal-close"));
      expect(onClose).toHaveBeenCalledTimes(1);
    });

    it("shows a full-width line on the overflow edge as the list scrolls", () => {
      renderWithProviders(
        <AdvancedConversationOptionsModal
          open
          onClose={vi.fn()}
          backendKind="local"
          automationNameFacets={[]}
        />,
      );

      const scroller = screen.getByTestId("advanced-options-scroll");
      const topEdge = screen.getByTestId("advanced-options-scroll-edge-top");
      const bottomEdge = screen.getByTestId(
        "advanced-options-scroll-edge-bottom",
      );

      mockVerticalScrollMetrics(scroller, {
        scrollHeight: 900,
        clientHeight: 320,
        scrollTop: 0,
      });
      fireEvent.scroll(scroller);

      expect(topEdge).toHaveAttribute("data-visible", "false");
      expect(bottomEdge).toHaveAttribute("data-visible", "true");

      mockVerticalScrollMetrics(scroller, {
        scrollHeight: 900,
        clientHeight: 320,
        scrollTop: 280,
      });
      fireEvent.scroll(scroller);

      expect(topEdge).toHaveAttribute("data-visible", "true");
      expect(bottomEdge).toHaveAttribute("data-visible", "true");

      mockVerticalScrollMetrics(scroller, {
        scrollHeight: 900,
        clientHeight: 320,
        scrollTop: 580,
      });
      fireEvent.scroll(scroller);

      expect(topEdge).toHaveAttribute("data-visible", "true");
      expect(bottomEdge).toHaveAttribute("data-visible", "false");
    });
  });
});

describe("BudgetDisplay", () => {
  it("renders nothing when cost is null", () => {
    const { container } = renderWithProviders(
      <BudgetDisplay cost={null} maxBudgetPerTask={5} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when maxBudgetPerTask is null (no budget cap)", () => {
    // Per OHE-3110: there is no way for a user to set max_budget_per_task, so
    // the "No budget limit" line must be omitted rather than shown.
    const { container } = renderWithProviders(
      <BudgetDisplay cost={1.23} maxBudgetPerTask={null} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when maxBudgetPerTask is zero", () => {
    const { container } = renderWithProviders(
      <BudgetDisplay cost={1.23} maxBudgetPerTask={0} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when maxBudgetPerTask is negative", () => {
    const { container } = renderWithProviders(
      <BudgetDisplay cost={1.23} maxBudgetPerTask={-1} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the progress bar and usage text when a real cap exists", () => {
    const { container } = renderWithProviders(
      <BudgetDisplay cost={1.23} maxBudgetPerTask={5} />,
    );
    // Cap-present branch renders two child rows: BudgetProgressBar (whose
    // fill div carries the transition-all class) and BudgetUsageText (the
    // usage key renders untranslated in tests). Asserting on both proves we
    // did not take the cap-absent (null) branch.
    expect(container.querySelector(".transition-all")).toBeInTheDocument();
    expect(
      screen.getByText("CONVERSATION$BUDGET_USAGE_FORMAT"),
    ).toBeInTheDocument();
  });
});

describe("budget-usage-text", () => {
  const FORMAT = translationResources.en.CONVERSATION$BUDGET_USAGE_FORMAT;

  // The exact parameters BudgetUsageText passes to t() (see
  // src/components/features/conversation-panel/budget-usage-text.tsx).
  const CONSUMER_PARAMS = {
    currentCost: "$0.1234",
    maxBudget: "$5.0000",
    usagePercentage: "2.47",
    used: translationResources.en.CONVERSATION$USED,
  };

  describe(I18nKey.CONVERSATION$BUDGET_USAGE_FORMAT, () => {
    it("uses i18next placeholder syntax in the en locale string", () => {
      // Regression guard for #16873: the key must use {{param}} placeholders so
      // i18next substitutes values instead of printing the literal template.
      expect(FORMAT).toBe(
        "{{currentCost}} / {{maxBudget}} ({{usagePercentage}}% {{used}})",
      );
      expect(FORMAT).not.toContain("${");
    });

    it("interpolates every named parameter the consumer passes", () => {
      for (const param of Object.keys(CONSUMER_PARAMS)) {
        expect(FORMAT).toContain(`{{${param}}}`);
      }
    });

    it("renders a fully substituted usage line with i18next", () => {
      // Drive a real i18next instance with the actual en resource to prove the
      // template interpolates end-to-end (the app's interpolation config matches:
      // default delimiters, escapeValue false).
      const instance = i18next.createInstance();
      instance.init({
        lng: "en",
        resources: { en: { openhands: translationResources.en } },
        ns: ["openhands"],
        defaultNS: "openhands",
        interpolation: { escapeValue: false },
      });

      const rendered = instance.t(
        I18nKey.CONVERSATION$BUDGET_USAGE_FORMAT,
        CONSUMER_PARAMS,
      ) as string;

      expect(rendered).toBe("$0.1234 / $5.0000 (2.47% used)");
      expect(rendered).not.toContain("${");
      expect(rendered).not.toContain("{{");
    });
  });
});

describe("ConfirmStopModal", () => {
  it("renders buttons with cancel-button and confirm-button test ids", () => {
    // Arrange: render the modal.
    renderWithProviders(
      <ConfirmStopModal onConfirm={vi.fn()} onCancel={vi.fn()} />,
    );

    // Assert: buttons expose the expected data-testid attributes.
    expect(screen.getByTestId("cancel-button")).toBeInTheDocument();
    expect(screen.getByTestId("confirm-button")).toBeInTheDocument();
  });
});

describe("conversation-active-tag-filters", () => {
  const renderStrip = (
    props: Partial<
      React.ComponentProps<typeof ConversationActiveTagFilters>
    > = {},
  ) =>
    renderWithProviders(
      <ConversationActiveTagFilters
        selectedFacets={[]}
        onToggleFacet={vi.fn()}
        selectedAutomationNames={[]}
        onToggleAutomationName={vi.fn()}
        onClearAll={vi.fn()}
        {...props}
      />,
    );

  describe("ConversationActiveTagFilters", () => {
    it("renders nothing when no filter is active", () => {
      renderStrip();

      expect(
        screen.queryByTestId("conversation-active-tag-filters"),
      ).not.toBeInTheDocument();
    });

    it("names every active facet so a narrowed list is never unexplained", () => {
      renderStrip({ selectedFacets: ["project=vault", "work"] });

      expect(
        screen.getByTestId("conversation-active-tag-filters"),
      ).toBeInTheDocument();
      // Keyed and bare tags both read the way the facet rows label them.
      expect(
        screen.getByTestId("active-tag-filter-project=vault"),
      ).toHaveTextContent("project=vault");
      expect(screen.getByTestId("active-tag-filter-work")).toHaveTextContent(
        "work",
      );
    });

    it("drops a single filter from the strip itself", async () => {
      const user = userEvent.setup();
      const onToggleFacet = vi.fn();
      renderStrip({ selectedFacets: ["project=vault", "work"], onToggleFacet });

      await user.click(screen.getByTestId("active-tag-filter-project=vault"));

      expect(onToggleFacet).toHaveBeenCalledWith("project=vault");
    });

    it("clears every filter at once", async () => {
      const user = userEvent.setup();
      const onClearAll = vi.fn();
      renderStrip({ selectedFacets: ["project=vault"], onClearAll });

      await user.click(screen.getByTestId("clear-tag-filters"));

      expect(onClearAll).toHaveBeenCalledTimes(1);
    });

    it("names an active automation-name filter too, so it is never the invisible narrowing", () => {
      // The automation-name selection is persisted and its rows live inside the
      // advanced-options modal; without a chip here a reload silently hides
      // conversations with nothing on screen to say why.
      renderStrip({ selectedAutomationNames: ["Nightly Audit"] });

      expect(
        screen.getByTestId("conversation-active-tag-filters"),
      ).toBeInTheDocument();
      expect(
        screen.getByTestId("active-automation-filter-Nightly Audit"),
      ).toHaveTextContent("Nightly Audit");
    });

    it("labels the unnamed-automation bucket rather than leaking its sentinel", () => {
      renderStrip({ selectedAutomationNames: [UNNAMED_AUTOMATION_FACET] });

      expect(
        screen.getByTestId(
          `active-automation-filter-${UNNAMED_AUTOMATION_FACET}`,
        ),
      ).not.toHaveTextContent(UNNAMED_AUTOMATION_FACET);
    });

    it("drops a single automation-name filter from the strip", async () => {
      const user = userEvent.setup();
      const onToggleAutomationName = vi.fn();
      renderStrip({
        selectedAutomationNames: ["Nightly Audit"],
        onToggleAutomationName,
      });

      await user.click(
        screen.getByTestId("active-automation-filter-Nightly Audit"),
      );

      expect(onToggleAutomationName).toHaveBeenCalledWith("Nightly Audit");
    });

    it("shows both families side by side", () => {
      renderStrip({
        selectedFacets: ["project=vault"],
        selectedAutomationNames: ["Nightly Audit"],
      });

      expect(
        screen.getByTestId("active-tag-filter-project=vault"),
      ).toBeInTheDocument();
      expect(
        screen.getByTestId("active-automation-filter-Nightly Audit"),
      ).toBeInTheDocument();
    });
  });
});
