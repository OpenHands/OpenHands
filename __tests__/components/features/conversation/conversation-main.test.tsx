import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { SidebarMobileNavProvider } from "#/components/features/sidebar/sidebar-mobile-nav-context";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
import type { AutomationSetupDraft } from "#/api/automation-setup-draft-store";

// Mutable mock state for controlling breakpoint
let mockIsMobile = false;
let mockIsRightPanelShown = false;
let mockLeftWidth = 50;
let mockAutomationSetupDraft: AutomationSetupDraft | null = null;

const mockNavigate = vi.fn();
const mockSetHasRightPanelToggled = vi.fn();
const mockSetIsRightPanelShown = vi.fn();
const mockClearAutomationSetupDraft = vi.fn();

// Track ChatInterface unmount via vi.fn()
const chatInterfaceUnmount = vi.fn();

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

vi.mock("#/hooks/use-breakpoint", () => ({
  useBreakpoint: () => mockIsMobile,
  SIDEBAR_RAIL_COLLAPSE_MAX_WIDTH: 767,
}));

vi.mock("#/hooks/use-resizable-panels", () => ({
  useResizablePanels: () => ({
    leftWidth: mockLeftWidth,
    rightWidth: 100 - mockLeftWidth,
    isDragging: false,
    containerRef: { current: null },
    handleMouseDown: vi.fn(),
  }),
}));

vi.mock("#/stores/conversation-store", () => ({
  useConversationStore: () => ({
    isRightPanelShown: mockIsRightPanelShown,
    setHasRightPanelToggled: mockSetHasRightPanelToggled,
    setIsRightPanelShown: mockSetIsRightPanelShown,
  }),
}));

vi.mock("#/api/automation-setup-draft-store", () => ({
  getAutomationSetupDraft: () => mockAutomationSetupDraft,
  subscribeAutomationSetupDraft: () => vi.fn(),
  clearAutomationSetupDraft: (...args: unknown[]) =>
    mockClearAutomationSetupDraft(...args),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: { title: "Daily Morning Haiku" },
  }),
}));

// Mock ChatInterface with useEffect to track mount/unmount lifecycle
vi.mock("#/components/features/chat/chat-interface", () => {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const React = require("react");
  return {
    ChatInterface: () => {
      React.useEffect(() => {
        return () => chatInterfaceUnmount();
      }, []);
      return <div data-testid="chat-interface">Chat Interface</div>;
    },
  };
});

vi.mock(
  "#/components/features/conversation/conversation-tabs/conversation-tab-content/conversation-tab-content",
  () => ({
    ConversationTabContent: () => <div data-testid="tab-content" />,
  }),
);

// ConversationMain now renders the conversation name and tabs inline as the
// pane headers; both reach into route/store state we don't set up here, so
// stub them out for layout-stability tests.
vi.mock(
  "#/components/features/conversation/conversation-name-with-status",
  () => ({
    ConversationNameWithStatus: () => (
      <div data-testid="conversation-name-with-status" />
    ),
  }),
);

vi.mock(
  "#/components/features/conversation/conversation-tabs/conversation-tabs",
  () => ({
    ConversationTabs: () => <div data-testid="conversation-tabs" />,
  }),
);

vi.mock(
  "#/components/features/automations/setup/automation-setup-panel",
  () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { createPortal } = require("react-dom");
    return {
      AutomationSetupPanel: ({
        toolbarPortal,
      }: {
        toolbarPortal?: HTMLElement | null;
      }) => (
        <>
          {toolbarPortal
            ? createPortal(
                <button type="button" data-testid="automation-setup-create">
                  Create automation
                </button>,
                toolbarPortal,
              )
            : null}
          <div data-testid="automation-setup-panel" />
        </>
      ),
    };
  },
);

import { ConversationMain } from "#/components/features/conversation/conversation-main/conversation-main";

function renderConversationMain() {
  const navigation: NavigationContextValue = {
    currentPath: "/conversations/conv-1",
    conversationId: "conv-1",
    isNavigating: false,
    navigate: mockNavigate,
  };

  return render(
    <NavigationProvider value={navigation}>
      <SidebarMobileNavProvider>
        <ConversationMain />
      </SidebarMobileNavProvider>
    </NavigationProvider>,
  );
}

describe("ConversationMain - Layout Transition Stability", () => {
  beforeEach(() => {
    mockIsMobile = false;
    mockIsRightPanelShown = false;
    mockLeftWidth = 50;
    mockAutomationSetupDraft = null;
    chatInterfaceUnmount.mockClear();
    mockNavigate.mockClear();
    mockSetHasRightPanelToggled.mockClear();
    mockSetIsRightPanelShown.mockClear();
    mockClearAutomationSetupDraft.mockClear();
  });

  it("renders ChatInterface at desktop width", () => {
    mockIsMobile = false;
    renderConversationMain();
    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });

  it("renders ChatInterface at mobile width", () => {
    mockIsMobile = true;
    renderConversationMain();
    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });

  it("does not unmount ChatInterface when crossing from desktop to mobile", () => {
    mockIsMobile = false;
    const { rerender } = renderConversationMain();
    expect(chatInterfaceUnmount).not.toHaveBeenCalled();

    // Cross the breakpoint to mobile
    mockIsMobile = true;
    rerender(
      <NavigationProvider
        value={{
          currentPath: "/conversations/conv-1",
          conversationId: "conv-1",
          isNavigating: false,
          navigate: mockNavigate,
        }}
      >
        <SidebarMobileNavProvider>
          <ConversationMain />
        </SidebarMobileNavProvider>
      </NavigationProvider>,
    );

    // ChatInterface must NOT have been unmounted and remounted
    expect(chatInterfaceUnmount).not.toHaveBeenCalled();
    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });

  it("does not unmount ChatInterface when crossing from mobile to desktop", () => {
    mockIsMobile = true;
    const { rerender } = renderConversationMain();
    expect(chatInterfaceUnmount).not.toHaveBeenCalled();

    // Cross the breakpoint to desktop
    mockIsMobile = false;
    rerender(
      <NavigationProvider
        value={{
          currentPath: "/conversations/conv-1",
          conversationId: "conv-1",
          isNavigating: false,
          navigate: mockNavigate,
        }}
      >
        <SidebarMobileNavProvider>
          <ConversationMain />
        </SidebarMobileNavProvider>
      </NavigationProvider>,
    );

    // ChatInterface must NOT have been unmounted and remounted
    expect(chatInterfaceUnmount).not.toHaveBeenCalled();
    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });

  it("survives rapid back-and-forth resize without unmounting ChatInterface", () => {
    mockIsMobile = false;
    const { rerender } = renderConversationMain();

    // Simulate rapid resize back and forth across the breakpoint
    for (const mobile of [true, false, true, false, true]) {
      mockIsMobile = mobile;
      rerender(
        <NavigationProvider
          value={{
            currentPath: "/conversations/conv-1",
            conversationId: "conv-1",
            isNavigating: false,
            navigate: mockNavigate,
          }}
        >
          <SidebarMobileNavProvider>
            <ConversationMain />
          </SidebarMobileNavProvider>
        </NavigationProvider>,
      );
    }

    expect(chatInterfaceUnmount).not.toHaveBeenCalled();
    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });

  it("uses a single automation setup top bar with splash back navigation", async () => {
    const user = userEvent.setup();
    mockIsRightPanelShown = true;
    mockAutomationSetupDraft = {
      prompt: "Write a haiku each morning",
      kind: "prompt",
    };

    renderConversationMain();

    expect(screen.getByTestId("automation-setup-topbar")).toBeInTheDocument();
    expect(screen.queryByTestId("chat-pane-header")).not.toBeInTheDocument();
    expect(
      screen.getByTestId("automation-setup-conversation-title"),
    ).toHaveTextContent("Daily Morning Haiku");
    expect(screen.getByTestId("automation-setup-create")).toBeInTheDocument();

    await user.click(screen.getByTestId("automation-setup-back"));

    expect(mockNavigate).toHaveBeenCalledWith("/");
  });

  it("expands the automation form to full width when hiding the agent", async () => {
    const user = userEvent.setup();
    mockIsRightPanelShown = true;
    mockAutomationSetupDraft = {
      prompt: "Write a haiku each morning",
      kind: "prompt",
    };

    renderConversationMain();

    expect(screen.getByTestId("conversation-chat-panel")).toHaveStyle({
      width: "50%",
    });
    expect(screen.getByTestId("conversation-right-panel")).toHaveStyle({
      width: "50%",
    });

    await user.click(screen.getByTestId("automation-setup-agent-toggle"));

    expect(screen.getByTestId("conversation-chat-panel")).toHaveStyle({
      width: "0%",
    });
    expect(screen.getByTestId("conversation-right-panel")).toHaveStyle({
      width: "100%",
    });
  });
});
