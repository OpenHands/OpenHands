import { screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { createRoutesStub } from "react-router";
import {
  setupConversationPanelTest,
  cloudBackend,
} from "./conversation-panel-test-utils";
import { renderWithProviders } from "test-utils";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ConversationPanel } from "#/components/features/conversation-panel/conversation-panel";
import { useConversationPanelPreferencesStore } from "#/stores/conversation-panel-preferences-store";

describe("ConversationPanel preview tooltip placement", () => {
  setupConversationPanelTest();

  beforeEach(() => {
    // React Aria interaction modality requires a pointer event to activate hover interactions in JSDOM
    fireEvent.pointerMove(document.body);
  });

  it("opens tooltip with right-start placement in desktop mode (isDrawer=false)", async () => {
    useConversationPanelPreferencesStore.setState({ showHoverMetadata: true });
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-2" });

    const DesktopRouterStub = createRoutesStub([
      {
        Component: () => <ConversationPanel isDrawer={false} />,
        path: "/",
      },
      {
        Component: () => null,
        path: "/conversations/:conversationId",
      },
    ]);

    const user = userEvent.setup();
    renderWithProviders(
      <ActiveBackendProvider>
        <DesktopRouterStub />
      </ActiveBackendProvider>,
    );

    const cards = await screen.findAllByTestId("conversation-card");
    const link = cards[0].closest("a")!;

    await user.hover(link);

    const tooltip = await screen.findByRole("tooltip");
    expect(tooltip).toBeInTheDocument();
    expect(tooltip).toHaveAttribute("data-placement", "right-start");
    expect(screen.getByTestId("conversation-card-preview")).toBeInTheDocument();

    await user.unhover(link);
  });

  it("opens tooltip with bottom-start placement in phone drawer mode (isDrawer=true)", async () => {
    useConversationPanelPreferencesStore.setState({ showHoverMetadata: true });
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-2" });

    const DrawerRouterStub = createRoutesStub([
      {
        Component: () => <ConversationPanel isDrawer />,
        path: "/",
      },
      {
        Component: () => null,
        path: "/conversations/:conversationId",
      },
    ]);

    const user = userEvent.setup();
    renderWithProviders(
      <ActiveBackendProvider>
        <DrawerRouterStub />
      </ActiveBackendProvider>,
    );

    const cards = await screen.findAllByTestId("conversation-card");
    const link = cards[0].closest("a")!;

    await user.hover(link);

    const tooltip = await screen.findByRole("tooltip");
    expect(tooltip).toBeInTheDocument();
    expect(tooltip).toHaveAttribute("data-placement", "bottom-start");
    expect(screen.getByTestId("conversation-card-preview")).toBeInTheDocument();

    await user.unhover(link);
  });

  it("adapts tooltip placement to bottom-start when viewport width is below md breakpoint", async () => {
    useConversationPanelPreferencesStore.setState({ showHoverMetadata: true });
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-2" });

    const ResponsiveRouterStub = createRoutesStub([
      {
        Component: () => <ConversationPanel />,
        path: "/",
      },
      {
        Component: () => null,
        path: "/conversations/:conversationId",
      },
    ]);

    // Simulate mobile viewport width (e.g. iPhone 390px)
    const originalInnerWidth = window.innerWidth;
    Object.defineProperty(window, "innerWidth", {
      writable: true,
      configurable: true,
      value: 390,
    });
    window.dispatchEvent(new Event("resize"));

    try {
      const user = userEvent.setup();
      renderWithProviders(
        <ActiveBackendProvider>
          <ResponsiveRouterStub />
        </ActiveBackendProvider>,
      );

      const cards = await screen.findAllByTestId("conversation-card");
      const link = cards[0].closest("a")!;

      await user.hover(link);

      const tooltip = await screen.findByRole("tooltip");
      expect(tooltip).toBeInTheDocument();
      expect(tooltip).toHaveAttribute("data-placement", "bottom-start");
      expect(
        screen.getByTestId("conversation-card-preview"),
      ).toBeInTheDocument();

      await user.unhover(link);
    } finally {
      Object.defineProperty(window, "innerWidth", {
        writable: true,
        configurable: true,
        value: originalInnerWidth,
      });
      window.dispatchEvent(new Event("resize"));
    }
  });
});
