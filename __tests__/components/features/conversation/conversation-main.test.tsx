import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { SidebarMobileNavProvider } from "#/components/features/sidebar/sidebar-mobile-nav-context";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
import type { AutomationSetupDraft } from "#/api/automation-setup-draft-store";
import { requestAutomationSetupAgent } from "#/components/features/automations/setup/automation-setup-agent-request";

// Mutable mock state for controlling breakpoint
let mockIsMobile = false;
let mockIsRightPanelShown = false;
let mockLeftWidth = 50;
let mockAutomationSetupDraft: AutomationSetupDraft | null = null;
let mockActiveConversationTags: Record<string, string> | null = null;

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
    data: { title: "Daily Morning Haiku", tags: mockActiveConversationTags },
  }),
}));

// Mock ChatInterface with useEffect to track mount/unmount lifecycle
vi.mock("#/components/features/chat/chat-interface", () => {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const React = require("react");
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { createPortal } = require("react-dom");
  return {
    ChatInterface: ({
      showGitControlBar = true,
      showEmptyStateSuggestions = true,
      composerDockTarget = null,
      onDockedComposerSubmit,
      minimalComposer = false,
      composerPlaceholder,
    }: {
      showGitControlBar?: boolean;
      showEmptyStateSuggestions?: boolean;
      composerDockTarget?: HTMLElement | null;
      onDockedComposerSubmit?: () => void;
      minimalComposer?: boolean;
      composerPlaceholder?: string;
    }) => {
      React.useEffect(() => {
        return () => chatInterfaceUnmount();
      }, []);
      return (
        <>
          <div
            data-testid="chat-interface"
            data-show-git-control-bar={showGitControlBar ? "true" : "false"}
            data-show-empty-state-suggestions={
              showEmptyStateSuggestions ? "true" : "false"
            }
            data-minimal-composer={minimalComposer ? "true" : "false"}
            data-composer-placeholder={composerPlaceholder}
          />
          {composerDockTarget
            ? createPortal(
                <button
                  type="button"
                  data-testid="docked-composer-submit"
                  onClick={() => onDockedComposerSubmit?.()}
                />,
                composerDockTarget,
              )
            : null}
        </>
      );
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
                <button type="button" data-testid="automation-setup-create" />,
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
    mockActiveConversationTags = null;
    chatInterfaceUnmount.mockClear();
    mockNavigate.mockClear();
    mockSetHasRightPanelToggled.mockClear();
    mockSetIsRightPanelShown.mockClear();
    mockClearAutomationSetupDraft.mockClear();
  });

  it("opens automation setup mode when the conversation has a draft tag", async () => {
    mockActiveConversationTags = { automationsetup: "draft" };

    renderConversationMain();

    expect(
      await screen.findByTestId("automation-setup-panel"),
    ).toBeInTheDocument();
    expect(mockSetHasRightPanelToggled).toHaveBeenCalledWith(true);
    expect(mockSetIsRightPanelShown).toHaveBeenCalledWith(true);
  });

  it("renders ChatInterface at desktop width", () => {
    mockIsMobile = false;
    renderConversationMain();
    expect(screen.getByTestId("chat-interface")).toHaveAttribute(
      "data-show-git-control-bar",
      "true",
    );
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

  it("uses a single automation setup top bar", () => {
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
    expect(screen.getByTestId("chat-interface")).toHaveAttribute(
      "data-show-git-control-bar",
      "false",
    );
    expect(screen.getByTestId("chat-interface")).toHaveAttribute(
      "data-show-empty-state-suggestions",
      "false",
    );
    expect(screen.getByTestId("chat-interface")).toHaveAttribute(
      "data-composer-placeholder",
      "AUTOMATION_SETUP$CONVERSATION_COMPOSER_PLACEHOLDER",
    );
    expect(
      screen.queryByTestId("automation-setup-back"),
    ).not.toBeInTheDocument();
  });

  it("keeps the automation form on a narrow window and toggles to the styled conversation", async () => {
    const user = userEvent.setup();
    mockIsMobile = true;
    mockIsRightPanelShown = true;
    mockAutomationSetupDraft = {
      prompt: "Write a haiku each morning",
      kind: "prompt",
    };

    renderConversationMain();

    expect(screen.getByTestId("automation-setup-topbar")).toBeInTheDocument();
    expect(screen.getByTestId("automation-setup-panel")).toBeInTheDocument();
    expect(
      screen.getByTestId("automation-setup-docked-composer"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("automation-setup-mobile-view-form"),
    ).toHaveAttribute("aria-label", "AUTOMATION_SETUP$VIEW_FORM");
    expect(
      screen.getByTestId("automation-setup-mobile-view-conversation"),
    ).toHaveAttribute("aria-label", "AUTOMATION_SETUP$VIEW_CONVERSATION");
    expect(screen.queryByTestId("chat-pane-header")).not.toBeInTheDocument();
    expect(screen.getByTestId("conversation-chat-panel")).toHaveClass("hidden");

    await user.click(
      screen.getByTestId("automation-setup-mobile-view-conversation"),
    );

    expect(
      screen.queryByTestId("automation-setup-mobile-form"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-setup-docked-composer"),
    ).not.toBeInTheDocument();
    expect(screen.queryByTestId("chat-pane-header")).not.toBeInTheDocument();
    expect(screen.getByTestId("conversation-chat-panel")).not.toHaveClass(
      "hidden",
    );
    expect(screen.getByTestId("chat-interface")).toHaveAttribute(
      "data-show-git-control-bar",
      "false",
    );
    expect(screen.getByTestId("chat-interface")).toHaveAttribute(
      "data-show-empty-state-suggestions",
      "false",
    );
    expect(screen.getByTestId("chat-interface")).toHaveAttribute(
      "data-minimal-composer",
      "true",
    );
    expect(screen.getByTestId("automation-setup-topbar")).toHaveClass(
      "flex-col",
    );

    await user.click(screen.getByTestId("automation-setup-mobile-view-form"));

    expect(
      screen.getByTestId("automation-setup-mobile-form"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("automation-setup-docked-composer"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("docked-composer-submit"));

    expect(
      screen.queryByTestId("automation-setup-mobile-form"),
    ).not.toBeInTheDocument();

    await user.click(screen.getByTestId("automation-setup-mobile-view-form"));

    act(() => {
      requestAutomationSetupAgent();
    });

    expect(
      screen.queryByTestId("automation-setup-mobile-form"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("conversation-chat-panel")).not.toHaveClass(
      "hidden",
    );
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
    expect(
      screen.queryByTestId("automation-setup-docked-composer"),
    ).not.toBeInTheDocument();

    await user.click(screen.getByTestId("automation-setup-agent-toggle"));

    expect(screen.getByTestId("conversation-chat-panel")).toHaveStyle({
      width: "0%",
    });
    expect(screen.getByTestId("conversation-right-panel")).toHaveStyle({
      width: "100%",
    });
    expect(
      screen.getByTestId("automation-setup-docked-composer"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("automation-setup-docked-composer").parentElement
        ?.parentElement,
    ).toHaveClass(
      "px-5",
      "custom-scrollbar-always",
      "[scrollbar-gutter:stable]",
    );
    expect(screen.getByTestId("docked-composer-submit")).toBeInTheDocument();

    await user.click(screen.getByTestId("docked-composer-submit"));

    expect(screen.getByTestId("conversation-chat-panel")).toHaveStyle({
      width: "50%",
    });
    expect(
      screen.queryByTestId("automation-setup-docked-composer"),
    ).not.toBeInTheDocument();
  });

  it("reveals the agent when the setup form asks for help", async () => {
    const user = userEvent.setup();
    mockIsRightPanelShown = true;
    mockAutomationSetupDraft = {
      prompt: "Write a haiku each morning",
      kind: "prompt",
    };

    renderConversationMain();
    await user.click(screen.getByTestId("automation-setup-agent-toggle"));
    expect(screen.getByTestId("conversation-chat-panel")).toHaveStyle({
      width: "0%",
    });

    act(() => {
      requestAutomationSetupAgent();
    });

    expect(screen.getByTestId("conversation-chat-panel")).toHaveStyle({
      width: "50%",
    });
  });
});
