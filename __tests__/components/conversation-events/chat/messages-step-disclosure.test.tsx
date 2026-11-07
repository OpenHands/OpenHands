import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Messages } from "#/components/conversation-events/chat/messages";
import { AgentState } from "#/types/agent-state";
import {
  ACPToolCallEvent,
  ActionEvent,
  HookExecutionEvent,
  MessageEvent,
  ObservationEvent,
  OpenHandsEvent,
  SecurityRisk,
} from "#/types/agent-server/core";
import { ExecuteBashAction } from "#/types/agent-server/core/base/action";
import { ExecuteBashObservation } from "#/types/agent-server/core/base/observation";
import { createUserMessageEvent, renderWithProviders } from "test-utils";

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({ data: {} }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: { id: "conversation-1", conversation_url: "", session_api_key: null },
  }),
}));

const agent = vi.hoisted(() => ({ state: "running" }));

vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => ({ curAgentState: agent.state }),
  usePlanningAgentState: () => ({ isPlanningAgentRunning: false }),
}));

const action = (
  id: string,
  command: string,
): ActionEvent<ExecuteBashAction> => ({
  id,
  timestamp: new Date().toISOString(),
  source: "agent",
  thought: [],
  thinking_blocks: [],
  action: {
    kind: "ExecuteBashAction",
    command,
    is_input: false,
    timeout: null,
    reset: false,
  },
  tool_name: "execute_bash",
  tool_call_id: `call_${id}`,
  tool_call: {
    id: `call_${id}`,
    type: "function",
    function: { name: "execute_bash", arguments: JSON.stringify({ command }) },
  },
  llm_response_id: `response_${id}`,
  security_risk: SecurityRisk.LOW,
});

const observation = (
  from: ActionEvent<ExecuteBashAction>,
): ObservationEvent<ExecuteBashObservation> => ({
  id: `obs_${from.id}`,
  timestamp: new Date().toISOString(),
  source: "environment",
  tool_name: from.tool_name,
  tool_call_id: from.tool_call_id,
  action_id: from.id,
  observation: {
    kind: "ExecuteBashObservation",
    content: [{ type: "text", text: `output of ${from.action.command}` }],
    command: from.action.command,
    exit_code: 0,
    error: false,
    timeout: false,
    metadata: {} as never,
  },
});

// UI events are the chat as rendered: a finished action is replaced by its
// observation, the full history keeps both.
const renderChat = (ui: OpenHandsEvent[], all: OpenHandsEvent[]) =>
  renderWithProviders(<Messages messages={ui} allEvents={all} />);

const openSteps = () =>
  screen.queryAllByRole("button", { name: "BUTTON$COLLAPSE" });

describe("Messages: opened steps and groups survive live updates", () => {
  beforeEach(() => {
    agent.state = AgentState.RUNNING;
  });

  it("keeps a step open when its running action finishes", async () => {
    const user = userEvent.setup();
    const a1 = action("a1", "ls");
    const { rerender } = renderChat([a1], [a1]);

    // While the agent works, its stage is a group from the first step.
    await user.click(screen.getByTestId("event-group-toggle"));
    expect(
      screen.queryByTestId("generic-event-message-title"),
    ).not.toBeInTheDocument();

    // The running action is replaced by its observation.
    rerender(
      <Messages
        messages={[observation(a1)]}
        allEvents={[a1, observation(a1)]}
      />,
    );

    expect(
      screen.queryByTestId("generic-event-message-title"),
    ).not.toBeInTheDocument();
    // The finished step's output is what the open row now shows.
    expect(document.body.textContent).toContain("output of ls");
  });

  it("keeps an idle group and its opened step expanded as tools arrive", async () => {
    agent.state = AgentState.AWAITING_USER_INPUT;
    const user = userEvent.setup();
    const a1 = { ...action("a1", "ls"), summary: "Inspecting the workspace" };
    const a2 = action("a2", "pwd");
    const { rerender } = renderChat([observation(a1)], [a1, observation(a1)]);

    expect(screen.getByTestId("event-group-toggle")).toHaveTextContent(
      "Inspecting the workspace",
    );
    await user.click(screen.getByTestId("event-group-toggle"));
    expect(document.body.textContent).toContain("output of ls");

    // A second step arrives, so the first one becomes part of a group.
    rerender(
      <Messages
        messages={[observation(a1), a2]}
        allEvents={[a1, observation(a1), a2]}
      />,
    );

    expect(screen.getByTestId("event-group-toggle")).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    expect(openSteps()).toHaveLength(1);
    // The finished step's output is what the open row now shows.
    expect(document.body.textContent).toContain("output of ls");
  });

  it("adds new steps to an open group as closed rows", async () => {
    const user = userEvent.setup();
    const [a1, a2, a3] = [
      action("a1", "ls"),
      action("a2", "pwd"),
      action("a3", "whoami"),
    ];
    const done = [observation(a1), observation(a2)];
    const { rerender } = renderChat(done, [a1, a2, ...done]);

    await user.click(screen.getByTestId("event-group-toggle"));
    rerender(
      <Messages messages={[...done, a3]} allEvents={[a1, a2, ...done, a3]} />,
    );

    expect(screen.getByTestId("event-group-toggle")).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    expect(screen.getAllByTestId("generic-event-message-title")).toHaveLength(
      3,
    );
    expect(openSteps()).toHaveLength(0);
  });

  it("keeps a group closed once the user closes it, even with an open step inside", async () => {
    const user = userEvent.setup();
    const [a1, a2, a3] = [
      action("a1", "ls"),
      action("a2", "pwd"),
      action("a3", "whoami"),
    ];
    const done = [observation(a1), observation(a2)];
    const { rerender } = renderChat(done, [a1, a2, ...done]);

    await user.click(screen.getByTestId("event-group-toggle"));
    await user.click(
      screen.getAllByRole("button", { name: "BUTTON$EXPAND" })[0],
    );
    await user.click(screen.getByTestId("event-group-toggle"));
    rerender(
      <Messages messages={[...done, a3]} allEvents={[a1, a2, ...done, a3]} />,
    );

    expect(screen.getByTestId("event-group-toggle")).toHaveAttribute(
      "aria-expanded",
      "false",
    );
  });
});

describe("Messages: reasoning in a finished conversation", () => {
  beforeEach(() => {
    agent.state = AgentState.FINISHED;
  });

  it("reveals a reply's reasoning with one click on its Thinking group", async () => {
    const user = userEvent.setup();
    const reply: MessageEvent = {
      id: "reply",
      timestamp: new Date().toISOString(),
      source: "agent",
      llm_message: {
        role: "assistant",
        content: [{ type: "text", text: "All done." }],
        reasoning_content: "Checking the tests passed",
      },
      activated_skills: [],
      extended_content: [],
    };
    renderChat([reply], [reply]);

    await user.click(screen.getByTestId("event-group-toggle"));

    expect(screen.getByText("Checking the tests passed")).toBeInTheDocument();
    expect(
      screen.queryByTestId("collapsible-thinking-toggle"),
    ).not.toBeInTheDocument();
  });

  it("keeps an action's reasoning in its step's group, below its message", async () => {
    const user = userEvent.setup();
    const a1 = {
      ...action("a1", "pwd"),
      thought: [{ type: "text" as const, text: "Now the directory." }],
      reasoning_content: "Need the working directory",
    };
    renderChat([observation(a1)], [a1, observation(a1)]);

    await user.click(screen.getByTestId("event-group-toggle"));

    const group = screen.getByTestId("event-group-content");
    await user.click(within(group).getByTestId("collapsible-thinking-toggle"));
    expect(
      within(group).getByText("Need the working directory"),
    ).toBeInTheDocument();
  });
});

describe("Messages: live status at the end of the chat", () => {
  beforeEach(() => {
    agent.state = AgentState.RUNNING;
  });

  it("shows a Thinking line while the agent works and no stage has started", () => {
    const message = createUserMessageEvent("m1");
    renderChat([message], [message]);

    expect(screen.getByTestId("live-thinking")).toHaveTextContent(
      "ACTION_MESSAGE$THINK",
    );
  });

  it("makes the stage the agent is on a group from its first step", () => {
    const message = createUserMessageEvent("m1");
    const a1 = action("a1", "ls");
    renderChat([message, a1], [message, a1]);

    // The live group carries the status, so there is no separate line.
    expect(screen.getByTestId("event-group-toggle")).toHaveTextContent(
      /ACTION_MESSAGE\$RUN/,
    );
    expect(screen.queryByTestId("live-thinking")).not.toBeInTheDocument();
  });

  it("shows no live status once the agent stops", () => {
    agent.state = AgentState.AWAITING_USER_INPUT;
    const message = createUserMessageEvent("m1");
    const a1 = action("a1", "ls");
    const done = [message, observation(a1)];
    renderChat(done, [message, a1, observation(a1)]);

    expect(screen.queryByTestId("live-thinking")).not.toBeInTheDocument();
    expect(screen.getByTestId("event-group-toggle")).toBeInTheDocument();
    expect(screen.queryByTestId("spinner-icon")).not.toBeInTheDocument();
  });
});

it("preserves a failed ACP command's revealed output outside the group", async () => {
  const user = userEvent.setup();
  const event: ACPToolCallEvent = {
    kind: "ACPToolCallEvent",
    id: "acp-1",
    timestamp: new Date().toISOString(),
    source: "agent",
    tool_call_id: "acp-call",
    title: "Check layout",
    tool_kind: "execute",
    status: "in_progress",
    raw_input: { command: "check-layout" },
    raw_output: "Checking layout",
    content: null,
    is_error: false,
  };
  const { rerender } = renderChat([event], [event]);
  await user.click(screen.getByTestId("event-group-toggle"));
  expect(document.body.textContent).toContain("Checking layout");
  const failed = {
    ...event,
    status: "failed" as const,
    is_error: true,
    raw_output: "Layout check failed",
  };
  rerender(<Messages messages={[failed]} allEvents={[failed]} />);
  expect(screen.queryByTestId("event-group-toggle")).not.toBeInTheDocument();
  expect(document.body.textContent).toContain("Layout check failed");
});

it("keeps commands collapsed in a pre-opened session until explicitly opened", async () => {
  const { getActiveBackend } =
    await import("#/api/backend-registry/active-store");
  const { sessionActivityKey, useSessionActivityStore } =
    await import("#/stores/session-activity-store");
  const { backend, orgId } = getActiveBackend();
  const key = sessionActivityKey(backend.id, orgId, "test-conversation-id");
  const store = useSessionActivityStore.getState();
  store.begin(key, "initialize");
  store.observe(key, "WAITING_FOR_SANDBOX");
  store.observe(key, "READY");
  const session = useSessionActivityStore.getState().entries[key][0];
  store.complete(key, session.id);
  store.setOpen(key, session.id, true);
  const { rerender } = renderChat([], []);
  const header = screen.getByTestId("event-group-toggle");
  const a1 = action("first", "ls");
  const o1 = observation(a1);
  rerender(<Messages messages={[o1]} allEvents={[a1, o1]} />);
  expect(document.body.textContent).not.toContain("output of ls");
  await userEvent.click(
    within(screen.getByTestId("generic-event-message-title")).getByRole(
      "button",
      { name: "BUTTON$EXPAND" },
    ),
  );
  expect(document.body.textContent).toContain("output of ls");
  const a2 = action("second", "pwd");
  rerender(<Messages messages={[o1, a2]} allEvents={[a1, o1, a2]} />);
  expect(screen.getByTestId("event-group-toggle")).toBe(header);
  expect(document.body.textContent).toContain("output of ls");
  expect(
    within(screen.getByTestId("session-activity-details")).getByRole("button"),
  ).toBeVisible();
  await userEvent.click(header);
  expect(document.body.textContent).not.toContain("output of ls");
  await userEvent.click(header);
  expect(document.body.textContent).toContain("output of ls");
  act(() => useSessionActivityStore.setState({ entries: {} }));
});

it("keeps a single tool's timeout visible on its group heading", async () => {
  agent.state = AgentState.AWAITING_USER_INPUT;
  const a1 = action("timeout", "slow-check");
  const done = observation(a1);
  done.observation.timeout = true;
  done.observation.exit_code = -1;
  renderChat([done], [a1, done]);
  const heading = screen.getByTestId("event-group-toggle");
  expect(within(heading).getByTestId("status-icon")).toBeVisible();
  await userEvent.click(heading);
  expect(screen.getAllByTestId("status-icon")).toHaveLength(1);
  expect(document.body.textContent).toContain("output of slow-check");
});

it("keeps rejection visible when the single command is collapsed", () => {
  agent.state = AgentState.AWAITING_USER_INPUT;
  const a1 = action("rejected", "ls");
  const rejected = {
    id: "rejection",
    timestamp: new Date().toISOString(),
    source: "environment" as const,
    tool_name: a1.tool_name,
    tool_call_id: a1.tool_call_id,
    action_id: a1.id,
    rejection_reason: "User rejected the action",
  };
  renderChat([rejected], [a1, rejected]);
  expect(screen.getByTestId("event-group-toggle")).toHaveTextContent(
    "EVENT$REJECTED",
  );
});

it("keeps the command collapsed alongside reasoning, while retaining its descriptive group title", async () => {
  agent.state = AgentState.AWAITING_USER_INPUT;
  const a1 = {
    ...action("reasoned", "ls"),
    summary: "Inspecting the workspace",
    reasoning_content: "Review the layout first",
  };
  const done = observation(a1);
  renderChat([done], [a1, done]);
  const heading = screen.getByTestId("event-group-toggle");
  expect(heading).toHaveTextContent("Inspecting the workspace");
  await userEvent.click(heading);
  expect(screen.getByTestId("collapsible-thinking")).toBeVisible();
  expect(document.body.textContent).not.toContain("output of ls");
  await userEvent.click(
    within(screen.getByTestId("generic-event-message-title")).getByRole(
      "button",
      { name: "BUTTON$EXPAND" },
    ),
  );
  expect(document.body.textContent).toContain("output of ls");
});

const blockedHook = (): HookExecutionEvent => ({
  id: "blocked-hook",
  timestamp: new Date().toISOString(),
  kind: "HookExecutionEvent",
  source: "hook",
  hook_event_type: "PreToolUse",
  hook_command: "check-protected-branch",
  success: false,
  blocked: true,
  exit_code: 2,
  reason: "Push a branch instead of main",
  tool_name: "terminal",
  action_id: "blocked-action",
  message_id: null,
  stdout: "",
  stderr: "",
  error: null,
  additional_context: null,
  hook_input: null,
});

it("keeps a blocked hook inside a clean group heading and reveals its reason on demand", async () => {
  const user = userEvent.setup();
  const a = action("check", "git status");
  const hook = blockedHook();
  renderChat([observation(a), hook], [a, observation(a), hook]);

  expect(screen.queryByText("blocked")).not.toBeInTheDocument();
  await user.click(screen.getByTestId("event-group-toggle"));
  const group = screen.getByTestId("event-group-content");
  expect(within(group).getByText("blocked")).toBeVisible();
  expect(screen.queryByText(hook.reason!)).not.toBeInTheDocument();
  await user.click(within(group).getByText(/PreToolUse/));
  expect(within(group).getByText(hook.reason!)).toBeVisible();
});

it("keeps an explicitly opened hook visible as a group grows and reopens", async () => {
  const user = userEvent.setup();
  const a = action("check", "git status");
  const hook = blockedHook();
  const { rerender } = renderChat([a], [a]);
  await user.click(screen.getByTestId("event-group-toggle"));
  rerender(
    <Messages
      messages={[observation(a), hook]}
      allEvents={[a, observation(a), hook]}
    />,
  );
  expect(screen.getByTestId("event-group-content")).toHaveTextContent(
    "output of git status",
  );
  await user.click(screen.getByText(/PreToolUse/));

  const next = action("next", "git diff");
  rerender(
    <Messages
      messages={[observation(a), hook, next]}
      allEvents={[a, observation(a), hook, next]}
    />,
  );
  expect(screen.getByText(hook.reason!)).toBeVisible();
  await user.click(screen.getByTestId("event-group-toggle"));
  await user.click(screen.getByTestId("event-group-toggle"));
  expect(screen.getByText(hook.reason!)).toBeVisible();
});

it("opens a lone blocked hook directly and preserves it when another tool joins", async () => {
  const user = userEvent.setup();
  const hook = blockedHook();
  const { rerender } = renderChat([hook], [hook]);
  expect(screen.queryByTestId("event-group-toggle")).not.toBeInTheDocument();
  await user.click(screen.getByText(/PreToolUse/));
  expect(screen.getByText(hook.reason!)).toBeVisible();

  const next = action("next", "git diff");
  rerender(<Messages messages={[hook, next]} allEvents={[hook, next]} />);
  expect(screen.getByTestId("event-group-toggle")).toHaveAttribute(
    "aria-expanded",
    "true",
  );
  expect(
    within(screen.getByTestId("event-group-content")).getByText(hook.reason!),
  ).toBeVisible();
});

describe("Messages: reasoning group identity", () => {
  it("opens reasoning independently and removes it when history changes", async () => {
    agent.state = AgentState.AWAITING_USER_INPUT;
    const actions = Array.from({ length: 8 }, (_, i) => ({
      ...action(`old-${i}`, "pwd"),
      thought: [{ type: "text" as const, text: `Checking step ${i}` }],
      reasoning_content: `Reasoning for step ${i}`,
    }));
    const done = actions.flatMap((a, i) => [
      observation(a),
      createUserMessageEvent(`old-prompt-${i}`),
    ]);
    const { rerender } = renderChat(done, [...actions, ...done]);
    await userEvent
      .setup()
      .click(screen.getAllByTestId("event-group-toggle")[0]);
    expect(screen.getAllByTestId("event-group-content")).toHaveLength(1);

    const message = createUserMessageEvent("new-prompt");
    rerender(<Messages messages={[message]} allEvents={[message]} />);
    expect(screen.queryAllByTestId("event-group-toggle")).toHaveLength(0);
  });
});
