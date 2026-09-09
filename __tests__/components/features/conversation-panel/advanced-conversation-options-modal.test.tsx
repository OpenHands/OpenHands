import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, screen, within } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { AdvancedConversationOptionsModal } from "#/components/features/conversation-panel/advanced-conversation-options-modal";
import { useConversationPanelPreferencesStore } from "#/stores/conversation-panel-preferences-store";

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
    ).toHaveClass("w-[520px]");
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

    expect(screen.getByText("CONVERSATION_PANEL$THREADS")).toBeInTheDocument();
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
    const bottomEdge = screen.getByTestId("advanced-options-scroll-edge-bottom");

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
