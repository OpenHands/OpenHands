import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { HookEventItem } from "#/components/features/conversation-panel/hook-event-item";
import { HooksEmptyState } from "#/components/features/conversation-panel/hooks-empty-state";
import { HooksLoadingState } from "#/components/features/conversation-panel/hooks-loading-state";
import { HooksModalHeader } from "#/components/features/conversation-panel/hooks-modal-header";
import { HookEvent } from "#/api/conversation-service/v1-conversation-service.types";

// Mock react-i18next
vi.mock("react-i18next", async () => {
  const actual = await vi.importActual("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string, params?: Record<string, unknown>) => {
        const translations: Record<string, string> = {
          HOOKS_MODAL$TITLE: "Available Hooks",
          HOOKS_MODAL$HOOK_COUNT: `${params?.count ?? 0} hooks`,
          HOOKS_MODAL$EVENT_PRE_TOOL_USE: "Pre Tool Use",
          HOOKS_MODAL$EVENT_POST_TOOL_USE: "Post Tool Use",
          HOOKS_MODAL$EVENT_USER_PROMPT_SUBMIT: "User Prompt Submit",
          HOOKS_MODAL$EVENT_SESSION_START: "Session Start",
          HOOKS_MODAL$EVENT_SESSION_END: "Session End",
          HOOKS_MODAL$EVENT_STOP: "Stop",
          HOOKS_MODAL$MATCHER: "Matcher",
          HOOKS_MODAL$COMMANDS: "Commands",
          HOOKS_MODAL$TYPE: `Type: ${params?.type ?? ""}`,
          HOOKS_MODAL$TIMEOUT: `Timeout: ${params?.timeout ?? 0}s`,
          HOOKS_MODAL$ASYNC: "Async",
          COMMON$FETCH_ERROR: "Failed to fetch data",
          CONVERSATION$NO_HOOKS: "No hooks configured",
          BUTTON$REFRESH: "Refresh",
        };
        return translations[key] || key;
      },
      i18n: {
        changeLanguage: () => new Promise(() => {}),
      },
    }),
  };
});

describe("HooksLoadingState", () => {
  it("should render loading spinner", () => {
    render(<HooksLoadingState />);
    const spinner = document.querySelector(".animate-spin");
    expect(spinner).toBeInTheDocument();
  });
});

describe("HooksEmptyState", () => {
  it("should render no hooks message when not error", () => {
    render(<HooksEmptyState isError={false} />);
    expect(screen.getByText("No hooks configured")).toBeInTheDocument();
  });

  it("should render error message when isError is true", () => {
    render(<HooksEmptyState isError={true} />);
    expect(screen.getByText("Failed to fetch data")).toBeInTheDocument();
  });
});

describe("HooksModalHeader", () => {
  const defaultProps = {
    isAgentReady: true,
    isLoading: false,
    isRefetching: false,
    onRefresh: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render title", () => {
    render(<HooksModalHeader {...defaultProps} />);
    expect(screen.getByText("Available Hooks")).toBeInTheDocument();
  });

  it("should render refresh button when agent is ready", () => {
    render(<HooksModalHeader {...defaultProps} />);
    expect(screen.getByTestId("refresh-hooks")).toBeInTheDocument();
  });

  it("should not render refresh button when agent is not ready", () => {
    render(<HooksModalHeader {...defaultProps} isAgentReady={false} />);
    expect(screen.queryByTestId("refresh-hooks")).not.toBeInTheDocument();
  });

  it("should call onRefresh when refresh button is clicked", async () => {
    const user = userEvent.setup();
    const onRefresh = vi.fn();
    render(<HooksModalHeader {...defaultProps} onRefresh={onRefresh} />);

    await user.click(screen.getByTestId("refresh-hooks"));
    expect(onRefresh).toHaveBeenCalledTimes(1);
  });

  it("should disable refresh button when loading", () => {
    render(<HooksModalHeader {...defaultProps} isLoading={true} />);
    expect(screen.getByTestId("refresh-hooks")).toBeDisabled();
  });

  it("should disable refresh button when refetching", () => {
    render(<HooksModalHeader {...defaultProps} isRefetching={true} />);
    expect(screen.getByTestId("refresh-hooks")).toBeDisabled();
  });
});

describe("HookEventItem", () => {
  const mockHookEvent: HookEvent = {
    event_type: "stop",
    matchers: [
      {
        matcher: "*",
        hooks: [
          {
            type: "command",
            command: ".openhands/hooks/on_stop.sh",
            timeout: 30,
            async: true,
          },
        ],
      },
    ],
  };

  const defaultProps = {
    hookEvent: mockHookEvent,
    isExpanded: false,
    onToggle: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render event type label using i18n", () => {
    render(<HookEventItem {...defaultProps} />);
    expect(screen.getByText("Stop")).toBeInTheDocument();
  });

  it("should render hook count", () => {
    render(<HookEventItem {...defaultProps} />);
    expect(screen.getByText("1 hooks")).toBeInTheDocument();
  });

  it("should call onToggle when clicked", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();
    render(<HookEventItem {...defaultProps} onToggle={onToggle} />);

    await user.click(screen.getByRole("button"));
    expect(onToggle).toHaveBeenCalledWith("stop");
  });

  it("should show collapsed state by default", () => {
    render(<HookEventItem {...defaultProps} isExpanded={false} />);
    // Matcher content should not be visible when collapsed
    expect(screen.queryByText("*")).not.toBeInTheDocument();
  });

  it("should show expanded state with matcher content", () => {
    render(<HookEventItem {...defaultProps} isExpanded={true} />);
    // Matcher content should be visible when expanded
    expect(screen.getByText("*")).toBeInTheDocument();
  });

  it("should render async badge for async hooks", () => {
    render(<HookEventItem {...defaultProps} isExpanded={true} />);
    expect(screen.getByText("Async")).toBeInTheDocument();
  });

  it("should render different event types with correct i18n labels", () => {
    const eventTypes = [
      { type: "pre_tool_use", label: "Pre Tool Use" },
      { type: "post_tool_use", label: "Post Tool Use" },
      { type: "user_prompt_submit", label: "User Prompt Submit" },
      { type: "session_start", label: "Session Start" },
      { type: "session_end", label: "Session End" },
      { type: "stop", label: "Stop" },
    ];

    eventTypes.forEach(({ type, label }) => {
      const { unmount } = render(
        <HookEventItem
          {...defaultProps}
          hookEvent={{ ...mockHookEvent, event_type: type }}
        />,
      );
      expect(screen.getByText(label)).toBeInTheDocument();
      unmount();
    });
  });

  it("should fallback to event_type when no i18n key exists", () => {
    render(
      <HookEventItem
        {...defaultProps}
        hookEvent={{ ...mockHookEvent, event_type: "unknown_event" }}
      />,
    );
    expect(screen.getByText("unknown_event")).toBeInTheDocument();
  });

  it("should not crash when a matcher has undefined hooks", () => {
    const hookEventWithUndefinedHooks: HookEvent = {
      event_type: "stop",
      matchers: [
        {
          matcher: "*",
          hooks: undefined,
        },
      ],
    };

    expect(() =>
      render(
        <HookEventItem
          {...defaultProps}
          hookEvent={hookEventWithUndefinedHooks}
        />,
      ),
    ).not.toThrow();

    expect(screen.getByText("0 hooks")).toBeInTheDocument();
  });

  it("should not crash when a matcher has undefined hooks in expanded state", () => {
    const hookEventWithUndefinedHooks: HookEvent = {
      event_type: "stop",
      matchers: [
        {
          matcher: "*",
          hooks: undefined,
        },
      ],
    };

    expect(() =>
      render(
        <HookEventItem
          {...defaultProps}
          hookEvent={hookEventWithUndefinedHooks}
          isExpanded={true}
        />,
      ),
    ).not.toThrow();
  });

  it("should handle a mix of matchers with and without hooks", () => {
    const mixedHookEvent: HookEvent = {
      event_type: "pre_tool_use",
      matchers: [
        {
          matcher: "terminal",
          hooks: [
            {
              type: "command",
              command: "check.sh",
              timeout: 10,
            },
          ],
        },
        {
          matcher: "browser",
          hooks: undefined,
        },
      ],
    };

    expect(() =>
      render(
        <HookEventItem
          {...defaultProps}
          hookEvent={mixedHookEvent}
          isExpanded={true}
        />,
      ),
    ).not.toThrow();

    // Should count only the valid hooks
    expect(screen.getByText("1 hooks")).toBeInTheDocument();
  });
});
