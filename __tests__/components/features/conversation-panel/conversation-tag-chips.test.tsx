import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { ConversationTagChips } from "#/components/features/conversation-panel/conversation-card/conversation-tag-chips";

describe("ConversationTagChips", () => {
  const observedCallbacks: ResizeObserverCallback[] = [];

  beforeEach(() => {
    observedCallbacks.length = 0;
    vi.stubGlobal(
      "ResizeObserver",
      class {
        constructor(cb: ResizeObserverCallback) {
          observedCallbacks.push(cb);
        }

        observe() {
          const cb = observedCallbacks[observedCallbacks.length - 1];
          cb?.([], this as unknown as ResizeObserver);
        }

        disconnect() {}

        unobserve() {}
      },
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function stubWidths(containerWidth: number, chipWidth: number) {
    const row = screen.getByTestId("conversation-card-tag-row");
    Object.defineProperty(row, "clientWidth", {
      configurable: true,
      get: () => containerWidth,
    });

    const measure = row
      .closest('[data-testid="conversation-card-tag-chips"]')
      ?.querySelector('[aria-hidden="true"]') as HTMLElement;
    Array.from(measure.children).forEach((child) => {
      Object.defineProperty(child, "offsetWidth", {
        configurable: true,
        get: () => chipWidth,
      });
    });

    act(() => {
      for (const cb of observedCallbacks) {
        cb([], {} as ResizeObserver);
      }
    });
  }

  it("folds chips that do not fit into a +N popover with friendly labels", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ConversationTagChips
        tags={[
          ["origin", "slack"],
          ["env", "prod"],
          ["owner", "alice"],
        ]}
      />,
    );

    // Wide enough for one 40px chip + overflow reserve, not two.
    stubWidths(80, 40);

    await waitFor(() => {
      expect(screen.getAllByTestId("conversation-card-tag-chip")).toHaveLength(
        1,
      );
    });
    const visibleChip = screen.getByTestId("conversation-card-tag-chip");
    expect(visibleChip).toHaveTextContent("slack");
    // Tooltip uses the humanized label — "Origin", not the wire key and not
    // "Git" (origin names the source of the conversation, not a git fact).
    expect(visibleChip).toHaveAttribute("title", "Origin: slack");

    const overflow = screen.getByTestId("conversation-card-tag-overflow");
    expect(overflow).toHaveTextContent("+2");

    await user.click(overflow);

    const popover = screen.getByTestId(
      "conversation-card-tag-overflow-popover",
    );
    expect(popover.parentElement).toBe(document.body);
    const rows = within(popover).getAllByTestId(
      "conversation-card-tag-overflow-row",
    );
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveTextContent("prod");
    expect(rows[0]).toHaveTextContent("Env");
    expect(rows[1]).toHaveTextContent("Owner");
    expect(rows[1]).toHaveTextContent("alice");
  });

  it("opens the overflow popover without activating a wrapping link", async () => {
    const user = userEvent.setup();
    const onNavigate = vi.fn((event: React.MouseEvent) => {
      event.preventDefault();
    });

    renderWithProviders(
      // Conversation cards sit inside NavigationLink anchors; the +N control
      // must swallow the activation so the popover can open in place.

      <a href="/conversations/demo" onClick={onNavigate}>
        <ConversationTagChips
          tags={[
            ["origin", "slack"],
            ["env", "prod"],
            ["owner", "alice"],
          ]}
        />
      </a>,
    );

    stubWidths(80, 40);

    await waitFor(() => {
      expect(
        screen.getByTestId("conversation-card-tag-overflow"),
      ).toBeInTheDocument();
    });

    await user.click(screen.getByTestId("conversation-card-tag-overflow"));

    expect(
      screen.getByTestId("conversation-card-tag-overflow-popover"),
    ).toBeInTheDocument();
    expect(onNavigate).not.toHaveBeenCalled();
  });

  it("hides the overflow control when every chip fits", async () => {
    renderWithProviders(
      <ConversationTagChips
        tags={[
          ["origin", "slack"],
          ["owner", "alice"],
        ]}
      />,
    );

    stubWidths(200, 40);

    await waitFor(() => {
      expect(screen.getAllByTestId("conversation-card-tag-chip")).toHaveLength(
        2,
      );
    });
    expect(
      screen.queryByTestId("conversation-card-tag-overflow"),
    ).not.toBeInTheDocument();
  });

  it("emits chip icon test ids only for visible chips, not the measure row", async () => {
    // The off-screen measure row renders the same chips; tagging those too
    // would make `getByTestId` (singular) ambiguous for every tagged card.
    renderWithProviders(
      <ConversationTagChips
        tags={[
          ["origin", "slack"],
          ["owner", "alice"],
        ]}
      />,
    );

    stubWidths(200, 40);

    await waitFor(() => {
      expect(screen.getAllByTestId("conversation-card-tag-chip")).toHaveLength(
        2,
      );
    });
    expect(
      screen.getAllByTestId("conversation-card-tag-chip-icon"),
    ).toHaveLength(2);
  });

  it("keeps the overflow popover open across re-renders with unchanged tags", async () => {
    // Callers rebuild the tag array on every render, and the conversation
    // panel re-renders at least every 10s (`refetchInterval`). Resetting on
    // array identity would slam the popover shut under the user's cursor.
    const user = userEvent.setup();
    const { rerender } = renderWithProviders(
      <ConversationTagChips
        tags={[
          ["origin", "slack"],
          ["env", "prod"],
          ["owner", "alice"],
        ]}
      />,
    );

    stubWidths(80, 40);

    await waitFor(() => {
      expect(
        screen.getByTestId("conversation-card-tag-overflow"),
      ).toBeInTheDocument();
    });
    await user.click(screen.getByTestId("conversation-card-tag-overflow"));
    expect(
      screen.getByTestId("conversation-card-tag-overflow-popover"),
    ).toBeInTheDocument();

    // Same content, fresh array identity — exactly what the parent produces.
    rerender(
      <ConversationTagChips
        tags={[
          ["origin", "slack"],
          ["env", "prod"],
          ["owner", "alice"],
        ]}
      />,
    );

    expect(
      screen.getByTestId("conversation-card-tag-overflow-popover"),
    ).toBeInTheDocument();
  });

  it("closes the overflow popover when the tags themselves change", async () => {
    const user = userEvent.setup();
    const { rerender } = renderWithProviders(
      <ConversationTagChips
        tags={[
          ["origin", "slack"],
          ["env", "prod"],
          ["owner", "alice"],
        ]}
      />,
    );

    stubWidths(80, 40);

    await waitFor(() => {
      expect(
        screen.getByTestId("conversation-card-tag-overflow"),
      ).toBeInTheDocument();
    });
    await user.click(screen.getByTestId("conversation-card-tag-overflow"));
    expect(
      screen.getByTestId("conversation-card-tag-overflow-popover"),
    ).toBeInTheDocument();

    rerender(
      <ConversationTagChips
        tags={[
          ["origin", "api"],
          ["env", "prod"],
          ["owner", "alice"],
        ]}
      />,
    );

    expect(
      screen.queryByTestId("conversation-card-tag-overflow-popover"),
    ).not.toBeInTheDocument();
  });
});
