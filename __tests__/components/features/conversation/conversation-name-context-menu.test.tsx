import React from "react";
import { screen } from "@testing-library/react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "test-utils";
import { ConversationNameContextMenu } from "#/components/features/conversation/conversation-name-context-menu";

vi.mock("#/hooks/use-breakpoint", () => ({
  useBreakpoint: () => false,
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: {
      id: "bundled",
      name: "Bundled",
      host: "http://localhost:3000",
      apiKey: "",
      kind: "local",
    },
    orgId: null,
  }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      public: false,
    },
  }),
}));

function createAnchor(rect: Partial<DOMRect> = {}) {
  const anchor = document.createElement("button");
  anchor.dataset.testAnchor = "true";
  Object.defineProperty(anchor, "getBoundingClientRect", {
    value: () => ({
      x: 12,
      y: 20,
      top: 20,
      right: 112,
      bottom: 48,
      left: 12,
      width: 100,
      height: 28,
      toJSON: () => ({}),
      ...rect,
    }),
  });
  document.body.appendChild(anchor);
  return anchor;
}

// The `style.left/top/bottom` assertions in this file verify the numeric
// output of the portal positioning math (anchor rect → menu coordinates).
// They are functional logic checks, not visual styling assertions.
describe("ConversationNameContextMenu portal rendering", () => {
  afterEach(() => {
    document
      .querySelectorAll('[data-test-anchor="true"]')
      .forEach((anchor) => anchor.remove());
    vi.restoreAllMocks();
  });

  it("renders inline when no anchor ref is provided", () => {
    const { container } = renderWithProviders(
      <ConversationNameContextMenu onClose={vi.fn()} onRename={vi.fn()} />,
    );

    expect(
      container.contains(screen.getByTestId("conversation-name-context-menu")),
    ).toBe(true);
  });

  it("renders in a portal and positions below the anchor when provided", () => {
    const anchor = createAnchor();
    const { container } = renderWithProviders(
      <ConversationNameContextMenu
        onClose={vi.fn()}
        onRename={vi.fn()}
        anchorRef={{ current: anchor }}
      />,
    );

    const menu = screen.getByTestId("conversation-name-context-menu");
    expect(container.contains(menu)).toBe(false);

    const wrapper = menu.parentElement as HTMLDivElement;
    expect(wrapper.style.left).toBe("12px");
    expect(wrapper.style.top).toBe("56px");
  });

  it("positions above the anchor for top-aligned menus", () => {
    const anchor = createAnchor({ top: 120, bottom: 148 });
    Object.defineProperty(window, "innerHeight", {
      configurable: true,
      value: 800,
    });

    renderWithProviders(
      <ConversationNameContextMenu
        onClose={vi.fn()}
        onRename={vi.fn()}
        position="top"
        anchorRef={{ current: anchor }}
      />,
    );

    const wrapper = screen.getByTestId("conversation-name-context-menu")
      .parentElement as HTMLDivElement;
    expect(wrapper.style.bottom).toBe("688px");
  });

  it("repositions when the anchored element changes between renders", () => {
    const firstAnchor = createAnchor({ left: 12, top: 20, bottom: 48 });
    const secondAnchor = createAnchor({ left: 64, top: 80, bottom: 112 });
    const { rerender } = renderWithProviders(
      <ConversationNameContextMenu
        onClose={vi.fn()}
        onRename={vi.fn()}
        anchorRef={{ current: firstAnchor }}
      />,
    );

    rerender(
      <ConversationNameContextMenu
        onClose={vi.fn()}
        onRename={vi.fn()}
        anchorRef={{ current: secondAnchor }}
      />,
    );

    const wrapper = screen.getByTestId("conversation-name-context-menu")
      .parentElement as HTMLDivElement;
    expect(wrapper.style.left).toBe("64px");
    expect(wrapper.style.top).toBe("120px");
  });

  it("registers and cleans up resize/scroll listeners for anchored menus", () => {
    const addEventListenerSpy = vi.spyOn(window, "addEventListener");
    const removeEventListenerSpy = vi.spyOn(window, "removeEventListener");
    const anchor = createAnchor();

    const { unmount } = renderWithProviders(
      <ConversationNameContextMenu
        onClose={vi.fn()}
        onRename={vi.fn()}
        anchorRef={{ current: anchor }}
      />,
    );

    expect(addEventListenerSpy).toHaveBeenCalledWith(
      "resize",
      expect.any(Function),
    );
    expect(addEventListenerSpy).toHaveBeenCalledWith(
      "scroll",
      expect.any(Function),
      true,
    );

    unmount();

    expect(removeEventListenerSpy).toHaveBeenCalledWith(
      "resize",
      expect.any(Function),
    );
    expect(removeEventListenerSpy).toHaveBeenCalledWith(
      "scroll",
      expect.any(Function),
      true,
    );
  });

  it("returns null during static rendering before portal coordinates exist", () => {
    const html = renderToStaticMarkup(
      <ConversationNameContextMenu
        onClose={vi.fn()}
        onRename={vi.fn()}
        anchorRef={{ current: {} as HTMLElement }}
      />,
    );

    expect(html).toBe("");
  });
});
