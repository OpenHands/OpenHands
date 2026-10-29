import { fireEvent, screen, render } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderWithProviders } from "test-utils";
import { ChangeAgentContextMenu } from "#/components/features/chat/change-agent-context-menu";
import i18n from "i18next";
import userEvent from "@testing-library/user-event";
import { ModelMessages } from "#/components/features/chat/model-messages";
import { useModelStore } from "#/stores/model-store";
import { useFreeModelsStore } from "#/stores/free-models-store";
import { type ProfileInfo } from "#/api/profiles-service/profiles-service.api";
import { deriveLiveActivity } from "#/components/features/chat/typing-indicator";
import {
  SecurityRisk,
  type ActionEvent,
  type ObservationEvent,
  type UserRejectObservation,
} from "#/types/agent-server/core";
import { type Action } from "#/types/agent-server/core/base/action";
import { type ACPToolCallEvent } from "#/types/agent-server/core/events/acp-tool-call-event";
import { type OHEvent } from "#/stores/use-event-store";
import { WorkspaceModeSelector } from "#/components/features/chat/workspace-mode-selector";

describe("ChangeAgentContextMenu", () => {
  it("marks the active mode with hover background and a checkmark", () => {
    renderWithProviders(
      <ChangeAgentContextMenu
        activeMode="code"
        onClose={vi.fn()}
        onCodeClick={vi.fn()}
        onPlanClick={vi.fn()}
      />,
    );

    const codeOption = screen.getByTestId("code-option");
    const planOption = screen.getByTestId("plan-option");

    expect(codeOption).toHaveClass("bg-interactive-hover");
    expect(planOption).not.toHaveClass("bg-interactive-hover");
    expect(codeOption.querySelectorAll("svg")).toHaveLength(2);
    expect(planOption.querySelectorAll("svg")).toHaveLength(1);
  });

  it("shows plan as active when conversation mode is plan", () => {
    renderWithProviders(
      <ChangeAgentContextMenu
        activeMode="plan"
        onClose={vi.fn()}
        onCodeClick={vi.fn()}
        onPlanClick={vi.fn()}
      />,
    );

    const codeOption = screen.getByTestId("code-option");
    const planOption = screen.getByTestId("plan-option");

    expect(planOption).toHaveClass("bg-interactive-hover");
    expect(codeOption).not.toHaveClass("bg-interactive-hover");
    expect(planOption.querySelectorAll("svg")).toHaveLength(2);
  });

  it("calls onCodeClick and onClose when code is selected", () => {
    const onClose = vi.fn();
    const onCodeClick = vi.fn();

    renderWithProviders(
      <ChangeAgentContextMenu
        activeMode="plan"
        onClose={onClose}
        onCodeClick={onCodeClick}
        onPlanClick={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByTestId("code-option"));

    expect(onCodeClick).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe("model-messages", () => {
  const CONVERSATION_ID = "conv-1";

  const profiles: ProfileInfo[] = [
    {
      name: "haiku",
      model: "anthropic/claude-haiku-4-5",
      base_url: "https://llm.example.test",
      api_key_set: true,
    },
    {
      name: "gpt",
      model: "openai/gpt-5.1",
      base_url: null,
      api_key_set: false,
    },
  ];

  describe("ModelMessages", () => {
    beforeEach(() => {
      const resources = {
        MODEL$AVAILABLE_PROFILES: "Available profiles ({{count}})",
        MODEL$NO_SAVED_PROFILES: "No saved profiles",
        MODEL$NO_PROFILES_HINT:
          "Use the LLM settings page to create a profile, then run /model <name> to switch.",
        MODEL$USER_SWITCHED: "You switched models to <cmd>{{name}}</cmd>",
        MODEL$AGENT_SWITCHED: "Agent switched models to <cmd>{{name}}</cmd>",
        MODEL$ROUTER_SWITCHED:
          "Model router switched models to <cmd>{{name}}</cmd>",
        MODEL$AGENT_SWITCH_REASON: "Agent’s reason:",
        MODEL$ROUTER_SWITCH_REASON: "Model router’s reason:",
        MODEL$SWITCHED_TO_PROFILE: "ℹ️ Switched to profile <cmd>{{name}}</cmd>",
      };
      i18n.addResourceBundle("en", "translation", resources, true, true);
      i18n.addResourceBundle("en", "openhands", resources, true, true);
      useModelStore.setState({ entriesByConversation: {} });
      useFreeModelsStore.getState().setFlags({
        freeModels: new Set(),
        defaultModel: null,
      });
    });

    it("renders only entries anchored to the requested event", async () => {
      const user = userEvent.setup();
      useModelStore.getState().show(CONVERSATION_ID, "event-1", [profiles[0]]);
      useModelStore.getState().show(CONVERSATION_ID, "event-2", [profiles[1]]);

      renderWithProviders(
        <ModelMessages
          conversationId={CONVERSATION_ID}
          anchorEventId="event-1"
        />,
      );

      expect(screen.getByTestId("model-messages")).toBeInTheDocument();
      expect(screen.getByText("Available profiles (1)")).toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "BUTTON$EXPAND" }));

      const profileToggle = screen.getByRole("button", {
        name: "Toggle details for haiku",
      });
      expect(profileToggle).toHaveAttribute("aria-expanded", "false");
      expect(
        screen.queryByRole("button", { name: "Toggle details for gpt" }),
      ).not.toBeInTheDocument();

      await user.click(profileToggle);
      expect(profileToggle).toHaveAttribute("aria-expanded", "true");

      expect(
        screen.getByText(/model:\s+anthropic\/claude-haiku-4-5/),
      ).toBeInTheDocument();
      expect(
        screen.getByText(/base_url:\s+https:\/\/llm\.example\.test/),
      ).toBeInTheDocument();
      expect(screen.getByText(/api_key:\s+set/)).toBeInTheDocument();
    });

    it("renders empty-profile hints expanded by default", () => {
      useModelStore.getState().show(CONVERSATION_ID, null, []);

      renderWithProviders(
        <ModelMessages conversationId={CONVERSATION_ID} anchorEventId={null} />,
      );

      expect(screen.getByText("No saved profiles")).toBeInTheDocument();
      expect(
        screen.getByText(
          "Use the LLM settings page to create a profile, then run /model <name> to switch.",
        ),
      ).toBeInTheDocument();
    });

    it("renders switch confirmations for the matching anchor", () => {
      useModelStore
        .getState()
        .recordSwitch(CONVERSATION_ID, "event-1", "haiku");

      renderWithProviders(
        <ModelMessages
          conversationId={CONVERSATION_ID}
          anchorEventId="event-1"
        />,
      );

      expect(screen.getByTestId("model-messages")).toHaveTextContent(
        "You switched models to haiku",
      );
    });

    it("shows an agent's rationale immediately as plain text", () => {
      useModelStore.getState().recordSwitch(CONVERSATION_ID, null, "astra", {
        initiator: "agent",
        reason: "Careful audit <script>alert(1)</script>",
      });
      renderWithProviders(
        <ModelMessages conversationId={CONVERSATION_ID} anchorEventId={null} />,
      );
      expect(screen.getByTestId("model-messages")).toHaveTextContent(
        "Agent switched models to astra",
      );
      expect(screen.getByText("Agent’s reason:")).toBeVisible();
      expect(
        screen.getByText("Careful audit <script>alert(1)</script>"),
      ).toBeVisible();
      expect(document.querySelector("script")).toBeNull();
    });

    it("distinguishes router switches and omits unavailable reasons", () => {
      useModelStore.getState().recordSwitch(CONVERSATION_ID, null, "fast", {
        initiator: "router",
      });
      renderWithProviders(
        <ModelMessages conversationId={CONVERSATION_ID} anchorEventId={null} />,
      );
      expect(screen.getByTestId("model-messages")).toHaveTextContent(
        "Model router switched models to fast",
      );
      expect(screen.queryByText("Agent’s reason:")).not.toBeInTheDocument();
      expect(
        screen.queryByText("Model router’s reason:"),
      ).not.toBeInTheDocument();
    });

    it("shows the router's chosen class as its reason when present", () => {
      useModelStore.getState().recordSwitch(CONVERSATION_ID, null, "fast", {
        initiator: "router",
        reason: "model: fast",
      });
      renderWithProviders(
        <ModelMessages conversationId={CONVERSATION_ID} anchorEventId={null} />,
      );
      expect(screen.getByTestId("model-messages")).toHaveTextContent(
        "Model router switched models to fast",
      );
      expect(screen.getByText("Model router’s reason:")).toBeVisible();
      expect(screen.getByText("model: fast")).toBeVisible();
      expect(screen.queryByText("Agent’s reason:")).not.toBeInTheDocument();
    });

    it("labels a DB-flagged free OpenHands route in expanded profile diagnostics", async () => {
      const user = userEvent.setup();
      useFreeModelsStore.getState().setFlags({
        freeModels: new Set(["openhands/glm-5.2"]),
        defaultModel: "openhands/glm-5.2",
      });
      useModelStore.getState().show(CONVERSATION_ID, "event-1", [
        {
          name: "free",
          model: "openhands/glm-5.2",
          base_url: null,
          api_key_set: true,
        },
      ]);

      renderWithProviders(
        <ModelMessages
          conversationId={CONVERSATION_ID}
          anchorEventId="event-1"
        />,
      );

      await user.click(screen.getByRole("button", { name: "BUTTON$EXPAND" }));
      await user.click(
        screen.getByRole("button", { name: "Toggle details for free" }),
      );

      expect(
        screen.getByText(/model:\s+openhands\/glm-5\.2 \(free\)/),
      ).toBeInTheDocument();
    });
  });
});

describe("typing-indicator", () => {
  const makeActionEvent = (
    id: string,
    action: Action,
    summary?: string,
  ): ActionEvent => ({
    id,
    timestamp: `2026-07-27T18:00:${id.padStart(2, "0")}Z`,
    source: "agent",
    thought: [],
    thinking_blocks: [],
    action,
    tool_name: action.kind,
    tool_call_id: `tool-${id}`,
    tool_call: {
      id: `tool-${id}`,
      type: "function",
      function: {
        name: action.kind,
        arguments: "{}",
      },
    },
    llm_response_id: `response-${id}`,
    security_risk: SecurityRisk.LOW,
    summary,
  });

  const makeObservationEvent = (
    id: string,
    action: ActionEvent,
  ): ObservationEvent => ({
    id,
    timestamp: `2026-07-27T18:01:${id.padStart(2, "0")}Z`,
    source: "environment",
    tool_name: action.tool_name,
    tool_call_id: action.tool_call_id,
    action_id: action.id,
    observation: {
      kind: "TerminalObservation",
      content: [],
      command: "git status",
      exit_code: 0,
      is_error: false,
      timeout: false,
      metadata: {
        exit_code: 0,
        pid: 1,
        username: "openhands",
        hostname: "runtime",
        prefix: "",
        suffix: "",
        working_dir: "/workspace/project",
        py_interpreter_path: null,
      },
    },
  });

  const makeUserRejectObservation = (
    id: string,
    action: ActionEvent,
  ): UserRejectObservation => ({
    id,
    timestamp: `2026-07-27T18:01:${id.padStart(2, "0")}Z`,
    source: "environment",
    tool_name: action.tool_name,
    tool_call_id: action.tool_call_id,
    action_id: action.id,
    rejection_reason: "Rejected by user",
  });

  const makeACPEvent = (
    id: string,
    status: ACPToolCallEvent["status"],
  ): ACPToolCallEvent => ({
    id: `${id}-${status ?? "legacy"}`,
    timestamp: "2026-07-27T18:02:00Z",
    kind: "ACPToolCallEvent",
    source: "agent",
    tool_call_id: id,
    title: "Read README.md",
    status,
    tool_kind: "read",
    raw_input: null,
    raw_output: null,
    content: null,
    is_error: status === "failed",
  });

  describe("deriveLiveActivity", () => {
    it.each([
      {
        name: "terminal command",
        action: {
          kind: "TerminalAction",
          command: "git status",
          is_input: false,
          timeout: null,
          reset: false,
        } satisfies Action,
        expected: {
          kind: "translation",
          key: "ACTION_MESSAGE$RUN",
          values: { command: "git status" },
        },
      },
      {
        name: "file read",
        action: {
          kind: "FileEditorAction",
          command: "view",
          path: "README.md",
          file_text: null,
          old_str: null,
          new_str: null,
          insert_line: null,
          view_range: null,
        } satisfies Action,
        expected: {
          kind: "translation",
          key: "ACTION_MESSAGE$READ",
          values: { path: "README.md" },
        },
      },
      {
        name: "code search",
        action: {
          kind: "GrepAction",
          pattern: "VITE_",
          path: null,
          include: null,
        } satisfies Action,
        expected: {
          kind: "translation",
          key: "ACTION_MESSAGE$GREP",
          values: { pattern: "VITE_" },
        },
      },
    ])("describes an unresolved $name", ({ action, expected }) => {
      const event = makeActionEvent("1", action);

      expect(deriveLiveActivity([event])).toEqual(expected);
    });

    it("moves to the next unresolved regular action as observations arrive", () => {
      const first = makeActionEvent("1", {
        kind: "TerminalAction",
        command: "npm test",
        is_input: false,
        timeout: null,
        reset: false,
      });
      const second = makeActionEvent("2", {
        kind: "FileEditorAction",
        command: "view",
        path: "README.md",
        file_text: null,
        old_str: null,
        new_str: null,
        insert_line: null,
        view_range: null,
      });
      const secondObservation = makeObservationEvent("3", second);
      const firstObservation = makeObservationEvent("4", first);

      expect(deriveLiveActivity([first, second, secondObservation])).toEqual({
        kind: "translation",
        key: "ACTION_MESSAGE$RUN",
        values: { command: "npm test" },
      });
      expect(
        deriveLiveActivity([
          first,
          second,
          secondObservation,
          firstObservation,
        ]),
      ).toEqual({
        kind: "translation",
        key: "ACTION_MESSAGE$THINK",
        values: {},
      });
    });

    it("treats a rejected regular action as resolved", () => {
      const rejected = makeActionEvent("1", {
        kind: "TerminalAction",
        command: "rm -rf build",
        is_input: false,
        timeout: null,
        reset: false,
      });

      expect(
        deriveLiveActivity([
          rejected,
          makeUserRejectObservation("2", rejected),
        ]),
      ).toEqual({
        kind: "translation",
        key: "ACTION_MESSAGE$THINK",
        values: {},
      });
    });

    it("shows the next unresolved action after a newer action is rejected", () => {
      const remaining = makeActionEvent("1", {
        kind: "TerminalAction",
        command: "git status",
        is_input: false,
        timeout: null,
        reset: false,
      });
      const rejected = makeActionEvent("2", {
        kind: "TerminalAction",
        command: "npm publish",
        is_input: false,
        timeout: null,
        reset: false,
      });

      expect(
        deriveLiveActivity([
          remaining,
          rejected,
          makeUserRejectObservation("3", rejected),
        ]),
      ).toEqual({
        kind: "translation",
        key: "ACTION_MESSAGE$RUN",
        values: { command: "git status" },
      });
    });

    it.each([
      "completed",
      "failed",
      null,
    ] satisfies readonly ACPToolCallEvent["status"][])(
      "drops an ACP label when the same call becomes %s",
      (terminalStatus) => {
        const started = makeACPEvent("acp-read", "in_progress");

        expect(deriveLiveActivity([started])).toEqual({
          kind: "translation",
          key: "ACTION_MESSAGE$ACP_READ",
          values: { title: "README.md" },
        });
        expect(
          deriveLiveActivity([
            started,
            makeACPEvent("acp-read", terminalStatus),
          ]),
        ).toEqual({
          kind: "translation",
          key: "ACTION_MESSAGE$THINK",
          values: {},
        });
      },
    );

    it("falls back to Thinking for unsupported and planning-only actions", () => {
      const unsupported = makeActionEvent("1", {
        kind: "SwitchLLMAction",
        profile_name: "fast",
        reason: "Short task",
      });
      const planningOnly = {
        ...makeActionEvent("2", {
          kind: "TerminalAction",
          command: "git status",
          is_input: false,
          timeout: null,
          reset: false,
        }),
        isFromPlanningAgent: true,
      } satisfies OHEvent;

      expect(deriveLiveActivity([unsupported])).toEqual({
        kind: "translation",
        key: "ACTION_MESSAGE$THINK",
        values: {},
      });
      expect(deriveLiveActivity([planningOnly])).toEqual({
        kind: "translation",
        key: "ACTION_MESSAGE$THINK",
        values: {},
      });
      expect(deriveLiveActivity([])).toEqual({
        kind: "translation",
        key: "ACTION_MESSAGE$THINK",
        values: {},
      });
    });
  });
});

describe("WorkspaceModeSelector", () => {
  it("opens the menu and calls onChange when an option is selected", () => {
    const onChange = vi.fn();
    render(
      <WorkspaceModeSelector
        value="local_repo"
        options={["local_repo", "new_worktree"]}
        backendKind="local"
        onChange={onChange}
      />,
    );

    // Open the menu (trigger renders the active mode via WorkspaceModeIcon).
    fireEvent.click(screen.getByTestId("workspace-mode-selector"));
    fireEvent.click(
      screen.getByTestId("workspace-mode-selector-option-new_worktree"),
    );

    expect(onChange).toHaveBeenCalledWith("new_worktree");
  });
});
