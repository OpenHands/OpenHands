import React, { isValidElement } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type {
  ACPToolCallEvent,
  Action,
  ActionEvent,
  MessageEvent,
  Observation,
  ObservationEvent,
  OpenHandsEvent,
} from "#/types/agent-server/core";
import { SecurityRisk } from "#/types/agent-server/core";
import type { SkillReadyEvent } from "./create-skill-ready-event";
import { getEventContent } from "./get-event-content";

const mocks = vi.hoisted(() => ({
  exists: vi.fn<(key: string) => boolean>(),
  translate: vi.fn<(key: string) => string>(),
  getActionContent: vi.fn(),
  getObservationContent: vi.fn(),
  getACPToolCallContent: vi.fn(),
  getACPToolCallTitleKey: vi.fn(),
  stripRedundantTitlePrefix: vi.fn(),
  resolveVisualizerBody: vi.fn(),
}));

vi.mock("#/i18n", () => ({
  default: {
    exists: mocks.exists,
    t: mocks.translate,
  },
}));

vi.mock("./get-action-content", () => ({
  getActionContent: mocks.getActionContent,
}));

vi.mock("./get-observation-content", () => ({
  getObservationContent: mocks.getObservationContent,
}));

vi.mock("./get-acp-tool-call-content", () => ({
  getACPToolCallContent: mocks.getACPToolCallContent,
  getACPToolCallTitleKey: mocks.getACPToolCallTitleKey,
  stripRedundantTitlePrefix: mocks.stripRedundantTitlePrefix,
}));

vi.mock("../../../features/chat/tool-visualizers/dispatcher", () => ({
  resolveVisualizerBody: mocks.resolveVisualizerBody,
}));

vi.mock("../task-tracking/task-tracking-observation-content", () => ({
  TaskTrackingObservationContent: ({ event }: { event: ObservationEvent }) => (
    <div data-testid="task-tracking-details">{event.id}</div>
  ),
}));

const TIMESTAMP = "2026-01-02T03:04:05.000Z";

const actionOf = (kind: string, fields: Record<string, unknown> = {}): Action =>
  ({ kind, ...fields }) as unknown as Action;

const observationOf = (
  kind: string,
  fields: Record<string, unknown> = {},
): Observation => ({ kind, ...fields }) as unknown as Observation;

const createActionEvent = (
  action: Action,
  overrides: Partial<ActionEvent> = {},
): ActionEvent => ({
  id: "action-event",
  timestamp: TIMESTAMP,
  source: "agent",
  thought: [],
  thinking_blocks: [],
  action,
  tool_name: "tool-name",
  tool_call_id: "tool-call",
  tool_call: {
    id: "tool-call",
    type: "function",
    function: { name: "tool-name", arguments: "{}" },
  },
  llm_response_id: "llm-response",
  security_risk: SecurityRisk.UNKNOWN,
  ...overrides,
});

const createObservationEvent = (
  observation: Observation,
  overrides: Partial<ObservationEvent> = {},
): ObservationEvent => ({
  id: "observation-event",
  timestamp: TIMESTAMP,
  source: "environment",
  tool_name: "tool-name",
  tool_call_id: "tool-call",
  action_id: "action-event",
  observation,
  ...overrides,
});

const createMessageEvent = (
  overrides: Partial<MessageEvent> = {},
): MessageEvent => ({
  id: "message-event",
  timestamp: TIMESTAMP,
  source: "user",
  llm_message: {
    role: "user",
    content: [{ type: "text", text: "hello" }],
  },
  activated_skills: [],
  extended_content: [],
  ...overrides,
});

const createSkillReadyEvent = (): SkillReadyEvent => ({
  id: "skill-ready-event",
  timestamp: TIMESTAMP,
  source: "agent",
  _isSkillReadyEvent: true,
  _skillReadyContent: "skill details",
  _skillReadyItems: [],
});

const createACPEvent = (): ACPToolCallEvent => ({
  id: "acp-event",
  timestamp: TIMESTAMP,
  source: "agent",
  kind: "ACPToolCallEvent",
  tool_call_id: "tool-call",
  title: "Read /workspace/file.ts",
  status: "completed",
  tool_kind: "read",
  raw_input: { path: "/workspace/file.ts" },
  raw_output: "contents",
  content: null,
  is_error: false,
});

type TranslationTitleProps = {
  i18nKey: string;
  values: Record<string, unknown>;
  components: Record<string, React.ReactNode>;
};

const getTranslationProps = (title: React.ReactNode): TranslationTitleProps => {
  if (!isValidElement<TranslationTitleProps>(title)) {
    throw new Error("Expected a translated React title");
  }

  return title.props;
};

const expectTranslatedTitle = (
  event: OpenHandsEvent | SkillReadyEvent,
  key: string,
  values: Record<string, unknown> = {},
  correspondingAction?: ActionEvent,
) => {
  const { title } = getEventContent(event, correspondingAction);
  const props = getTranslationProps(title);
  expect(props).toMatchObject({
    i18nKey: key,
    values,
  });
  expect(Object.keys(props.components)).toEqual(["path", "cmd"]);
  expect(isValidElement(props.components.path)).toBe(true);
  expect(isValidElement(props.components.cmd)).toBe(true);
};

beforeEach(() => {
  vi.clearAllMocks();
  mocks.exists.mockReturnValue(true);
  mocks.translate.mockImplementation((key) => `translated:${key}`);
  mocks.getActionContent.mockReturnValue("action details");
  mocks.getObservationContent.mockReturnValue("observation details");
  mocks.resolveVisualizerBody.mockReturnValue(null);
});

describe("getEventContent detail routing", () => {
  it("uses the translated skill-ready title when it exists", () => {
    const event = createSkillReadyEvent();

    const result = getEventContent(event);

    expect(getTranslationProps(result.title)).toMatchObject({
      i18nKey: "OBSERVATION_MESSAGE$SKILL_READY",
      values: {},
    });
    expect(result.details).toBe("skill details");
    expect(mocks.exists).toHaveBeenCalledTimes(2);
  });

  it("uses the English skill-ready fallback when the translation is absent", () => {
    mocks.exists.mockReturnValue(false);

    expect(getEventContent(createSkillReadyEvent())).toEqual({
      title: "Skill Ready",
      details: "skill details",
    });
  });

  it("prefers an action visualizer, including an intentionally empty body", () => {
    const event = createActionEvent(actionOf("ThinkAction", { thought: "x" }));
    mocks.resolveVisualizerBody.mockReturnValue("");

    const result = getEventContent(event);

    expect(result.details).toBe("");
    expect(mocks.resolveVisualizerBody).toHaveBeenCalledWith(event);
    expect(mocks.getActionContent).not.toHaveBeenCalled();
  });

  it("uses action markdown when no visualizer is registered", () => {
    const event = createActionEvent(actionOf("ThinkAction", { thought: "x" }));

    expect(getEventContent(event).details).toBe("action details");
    expect(mocks.getActionContent).toHaveBeenCalledWith(event);
  });

  it("prefers an observation visualizer and passes the matching action", () => {
    const action = createActionEvent(
      actionOf("ThinkAction", { thought: "reasoning" }),
    );
    const event = createObservationEvent(
      observationOf("ThinkObservation", { content: [] }),
    );
    mocks.resolveVisualizerBody.mockReturnValue("visualized observation");

    const result = getEventContent(event, action);

    expect(result.details).toBe("visualized observation");
    expect(mocks.resolveVisualizerBody).toHaveBeenCalledWith(event, action);
    expect(mocks.getObservationContent).not.toHaveBeenCalled();
  });

  it("uses observation markdown when no visualizer is registered", () => {
    const event = createObservationEvent(
      observationOf("ThinkObservation", { content: [] }),
    );

    expect(getEventContent(event).details).toBe("observation details");
    expect(mocks.getObservationContent).toHaveBeenCalledWith(event);
  });

  it("renders task-tracker details through its structured component", () => {
    const event = createObservationEvent(
      observationOf("TaskTrackerObservation", {
        command: "view",
        content: "one task",
        task_list: [],
      }),
    );

    const { details } = getEventContent(event);

    expect(isValidElement(details)).toBe(true);
    if (!isValidElement<{ event: ObservationEvent }>(details)) {
      throw new Error("Expected task-tracker React details");
    }
    expect(details.props.event).toBe(event);
    expect(mocks.resolveVisualizerBody).not.toHaveBeenCalled();
    expect(mocks.getObservationContent).not.toHaveBeenCalled();
  });

  it("delegates ACP titles and details to the ACP helpers", () => {
    const event = createACPEvent();
    mocks.getACPToolCallTitleKey.mockReturnValue("ACTION_MESSAGE$ACP_READ");
    mocks.stripRedundantTitlePrefix.mockReturnValue("/workspace/file.ts");
    mocks.getACPToolCallContent.mockReturnValue("ACP details");

    const result = getEventContent(event);

    expect(getTranslationProps(result.title)).toMatchObject({
      i18nKey: "ACTION_MESSAGE$ACP_READ",
      values: { title: "/workspace/file.ts" },
    });
    expect(result.details).toBe("ACP details");
    expect(mocks.getACPToolCallTitleKey).toHaveBeenCalledWith(event);
    expect(mocks.stripRedundantTitlePrefix).toHaveBeenCalledWith(event);
    expect(mocks.getACPToolCallContent).toHaveBeenCalledWith(event);
  });
});

describe("action titles", () => {
  it("uses a named auxiliary vision profile ahead of an LLM summary", () => {
    const event = createActionEvent(
      actionOf("MCPToolAction", {
        data: {},
        profile_name: "  fast vision  ",
      }),
      {
        tool_name: "inspect_image_with_vision",
        summary: "A less specific summary",
      },
    );

    expect(getEventContent(event).title).toBe(
      "Describing image with fast vision",
    );
  });

  it.each([
    ["missing", undefined],
    ["blank", "   "],
    ["non-string", 42],
  ])(
    "uses the auxiliary vision fallback for a %s profile",
    (_name, profile) => {
      const event = createActionEvent(
        actionOf("MCPToolAction", { data: {}, profile_name: profile }),
        { tool_name: "inspect_image_with_vision" },
      );

      expect(getEventContent(event).title).toBe(
        "Describing image with auxiliary vision LLM",
      );
    },
  );

  it("normalizes a meaningful LLM summary", () => {
    const event = createActionEvent(
      actionOf("ThinkAction", { thought: "reasoning" }),
      { summary: "  Inspecting\n   the result  " },
    );

    expect(getEventContent(event).title).toBe("Inspecting the result");
  });

  it("keeps a fallback-shaped fragment when it is not the whole summary", () => {
    const summary = 'Completed execute_bash: {"command":"npm test"}';
    const event = createActionEvent(
      actionOf("ThinkAction", { thought: "reasoning" }),
      { summary },
    );

    expect(getEventContent(event).title).toBe(summary);
  });

  it.each([
    'execute_bash: {"command":"npm test"}',
    'execute_bash:{"command":"npm test"}',
    'execute_bash : {"command":"npm test"}',
    'grep: ["one", "two"]',
    "   ",
  ])("ignores a non-human summary and uses the action title: %s", (summary) => {
    const event = createActionEvent(
      actionOf("ThinkAction", { thought: "reasoning" }),
      { summary },
    );

    expectTranslatedTitle(event, "ACTION_MESSAGE$THINK");
  });

  const longCommand = "c".repeat(81);
  const exactCommand = "e".repeat(80);
  const longPattern = "p".repeat(51);

  it.each([
    {
      name: "execute bash",
      action: actionOf("ExecuteBashAction", {
        command: longCommand,
        is_input: false,
        timeout: null,
        reset: false,
      }),
      key: "ACTION_MESSAGE$RUN",
      values: { command: `${"c".repeat(80)}...` },
    },
    {
      name: "terminal",
      action: actionOf("TerminalAction", {
        command: "pwd",
        is_input: false,
        timeout: null,
        reset: false,
      }),
      key: "ACTION_MESSAGE$RUN",
      values: { command: "pwd" },
    },
    {
      name: "exact-length terminal command",
      action: actionOf("TerminalAction", {
        command: exactCommand,
        is_input: false,
        timeout: null,
        reset: false,
      }),
      key: "ACTION_MESSAGE$RUN",
      values: { command: exactCommand },
    },
    {
      name: "empty terminal command",
      action: actionOf("TerminalAction", {
        command: "",
        is_input: true,
        timeout: null,
        reset: false,
      }),
      key: "ACTION_MESSAGE$RUN",
      values: { command: "" },
    },
    {
      name: "file read",
      action: actionOf("FileEditorAction", {
        command: "view",
        path: "/workspace/read.ts",
      }),
      key: "ACTION_MESSAGE$READ",
      values: { path: "/workspace/read.ts" },
    },
    {
      name: "file create",
      action: actionOf("FileEditorAction", {
        command: "create",
        path: "/workspace/create.ts",
      }),
      key: "ACTION_MESSAGE$WRITE",
      values: { path: "/workspace/create.ts" },
    },
    {
      name: "file edit",
      action: actionOf("FileEditorAction", {
        command: "str_replace",
        path: "/workspace/edit.ts",
      }),
      key: "ACTION_MESSAGE$EDIT",
      values: { path: "/workspace/edit.ts" },
    },
    {
      name: "legacy editor",
      action: actionOf("StrReplaceEditorAction", {
        command: "view",
        path: "/workspace/legacy.ts",
      }),
      key: "ACTION_MESSAGE$READ",
      values: { path: "/workspace/legacy.ts" },
    },
    {
      name: "MCP tool",
      action: actionOf("MCPToolAction", { data: {} }),
      toolName: "github_search",
      key: "ACTION_MESSAGE$CALL_TOOL_MCP",
      values: { mcp_tool_name: "github_search" },
    },
    {
      name: "skill",
      action: actionOf("InvokeSkillAction", { name: "testing" }),
      key: "ACTION_MESSAGE$INVOKE_SKILL",
      values: { name: "testing" },
    },
    {
      name: "subagent task",
      action: actionOf("TaskAction", {
        prompt: "inspect tests",
        subagent_type: "reviewer",
      }),
      key: "ACTION_MESSAGE$TASK",
      values: { name: "reviewer" },
    },
    {
      name: "thinking",
      action: actionOf("ThinkAction", { thought: "reasoning" }),
      key: "ACTION_MESSAGE$THINK",
      values: {},
    },
    {
      name: "finish",
      action: actionOf("FinishAction", { message: "done" }),
      key: "ACTION_MESSAGE$FINISH",
      values: {},
    },
    {
      name: "task tracking",
      action: actionOf("TaskTrackerAction", {
        command: "view",
        task_list: [],
      }),
      key: "ACTION_MESSAGE$TASK_TRACKING",
      values: {},
    },
    {
      name: "grep with a long pattern",
      action: actionOf("GrepAction", {
        pattern: longPattern,
        path: null,
        include: null,
      }),
      key: "ACTION_MESSAGE$GREP",
      values: { pattern: `${"p".repeat(50)}...` },
    },
    {
      name: "grep with an empty pattern",
      action: actionOf("GrepAction", {
        pattern: "",
        path: null,
        include: null,
      }),
      key: "ACTION_MESSAGE$GREP",
      values: { pattern: "" },
    },
    {
      name: "grep without a pattern field",
      action: actionOf("GrepAction", { path: null, include: null }),
      key: "ACTION_MESSAGE$GREP",
      values: { pattern: "" },
    },
    {
      name: "glob",
      action: actionOf("GlobAction", { pattern: "src/**", path: null }),
      key: "ACTION_MESSAGE$GLOB",
      values: { pattern: "src/**" },
    },
    {
      name: "empty glob",
      action: actionOf("GlobAction", { pattern: "", path: null }),
      key: "ACTION_MESSAGE$GLOB",
      values: { pattern: "" },
    },
    {
      name: "glob without a pattern field",
      action: actionOf("GlobAction", { path: null }),
      key: "ACTION_MESSAGE$GLOB",
      values: { pattern: "" },
    },
  ])("formats the $name action", ({ action, key, values, toolName }) => {
    const event = createActionEvent(action, {
      tool_name: toolName ?? "tool-name",
    });

    expectTranslatedTitle(event, key, values);
  });

  it.each([
    actionOf("BrowserNavigateAction", { url: "https://example.com" }),
    actionOf("BrowserClickAction", { index: 1, new_tab: false }),
    actionOf("BrowserTypeAction", { index: 1, text: "query" }),
    actionOf("BrowserGetStateAction", { include_screenshot: false }),
    actionOf("BrowserGetContentAction", {
      extract_links: false,
      start_from_char: 0,
    }),
    actionOf("BrowserScrollAction", { direction: "down" }),
    actionOf("BrowserGoBackAction"),
    actionOf("BrowserListTabsAction"),
    actionOf("BrowserSwitchTabAction", { tab_id: "abcd" }),
    actionOf("BrowserCloseTabAction", { tab_id: "abcd" }),
  ])("groups $kind under the browse title", (action) => {
    expectTranslatedTitle(createActionEvent(action), "ACTION_MESSAGE$BROWSE");
  });

  it("falls back to an uppercase name for an action without a title mapping", () => {
    const event = createActionEvent(
      actionOf("PlanningFileEditorAction", {
        command: "view",
        path: "/workspace/PLAN.md",
      }),
    );

    expect(getEventContent(event).title).toBe("PLANNINGFILEEDITOR");
  });

  it("shows the translation key when a mapped translation is unavailable", () => {
    mocks.exists.mockReturnValue(false);
    const event = createActionEvent(
      actionOf("ThinkAction", { thought: "reasoning" }),
    );

    expect(getEventContent(event).title).toBe("ACTION_MESSAGE$THINK");
  });
});

describe("observation titles", () => {
  it("uses the corresponding action's named profile for a vision result", () => {
    const action = createActionEvent(
      actionOf("MCPToolAction", {
        data: {},
        profile_name: "  vision pro  ",
      }),
      { tool_name: "inspect_image_with_vision" },
    );
    const event = createObservationEvent(
      observationOf("MCPToolObservation", {
        content: [],
        is_error: false,
        tool_name: "other-tool",
      }),
    );

    expect(getEventContent(event, action).title).toBe(
      "Describing image with vision pro",
    );
  });

  it("uses an observation profile when the action profile is blank", () => {
    const action = createActionEvent(
      actionOf("MCPToolAction", { data: {}, profile_name: "  " }),
      { tool_name: "inspect_image_with_vision" },
    );
    const event = createObservationEvent(
      observationOf("MCPToolObservation", {
        content: [],
        is_error: false,
        tool_name: "other-tool",
        profile_name: "observation profile",
      }),
    );

    expect(getEventContent(event, action).title).toBe(
      "Describing image with observation profile",
    );
  });

  it("detects a vision result from its tool name", () => {
    const event = createObservationEvent(
      observationOf("MCPToolObservation", {
        content: [],
        is_error: false,
        tool_name: " inspect_image_with_vision ",
        profile_name: "named profile",
      }),
    );

    expect(getEventContent(event).title).toBe(
      "Describing image with named profile",
    );
  });

  it("truncates a vision base URL to a compact label", () => {
    const baseUrl = `https://vision.example/${"x".repeat(80)}`;
    const event = createObservationEvent(
      observationOf("VisionInspectObservation", { base_url: baseUrl }),
    );

    expect(getEventContent(event).title).toBe(
      `Describing image with ${baseUrl.slice(0, 80)}...`,
    );
  });

  it.each([
    {
      name: "endpoint",
      fields: { base_url: " ", endpoint: "https://vision.example/v1" },
      expected: "https://vision.example/v1",
    },
    {
      name: "model",
      fields: { base_url: 42, endpoint: "", model: "vision-model" },
      expected: "vision-model",
    },
  ])(
    "uses the vision $name when earlier labels are absent",
    ({ fields, expected }) => {
      const event = createObservationEvent(
        observationOf("VisionInspectObservation", fields),
      );

      expect(getEventContent(event).title).toBe(
        `Describing image with ${expected}`,
      );
    },
  );

  it("uses the auxiliary vision fallback without any usable label", () => {
    const event = createObservationEvent(
      observationOf("VisionInspectObservation", {
        profile_name: null,
        base_url: " ",
        endpoint: 42,
        model: "",
      }),
    );

    expect(getEventContent(event).title).toBe(
      "Describing image with auxiliary vision LLM",
    );
  });

  it("uses a corresponding action summary for an ordinary observation", () => {
    const action = createActionEvent(
      actionOf("ThinkAction", { thought: "reasoning" }),
      { summary: "  Verified\n the output " },
    );
    const event = createObservationEvent(
      observationOf("ThinkObservation", { content: [] }),
    );

    expect(getEventContent(event, action).title).toBe("Verified the output");
  });

  it("ignores a corresponding action's server fallback summary", () => {
    const action = createActionEvent(
      actionOf("ThinkAction", { thought: "reasoning" }),
      { summary: 'think: {"thought":"reasoning"}' },
    );
    const event = createObservationEvent(
      observationOf("ThinkObservation", { content: [] }),
    );

    expectTranslatedTitle(event, "OBSERVATION_MESSAGE$THINK", {}, action);
  });

  const longCommand = "r".repeat(81);
  const longPattern = "g".repeat(51);

  it.each([
    {
      name: "execute bash",
      observation: observationOf("ExecuteBashObservation", {
        command: longCommand,
      }),
      key: "OBSERVATION_MESSAGE$RUN",
      values: { command: `${"r".repeat(80)}...` },
    },
    {
      name: "terminal without a command",
      observation: observationOf("TerminalObservation", { command: null }),
      key: "OBSERVATION_MESSAGE$RUN",
      values: { command: "" },
    },
    {
      name: "file read",
      observation: observationOf("FileEditorObservation", {
        command: "view",
        path: "/workspace/read.ts",
      }),
      key: "OBSERVATION_MESSAGE$READ",
      values: { path: "/workspace/read.ts" },
    },
    {
      name: "file edit without a path",
      observation: observationOf("FileEditorObservation", {
        command: "str_replace",
        path: null,
      }),
      key: "OBSERVATION_MESSAGE$EDIT",
      values: { path: "" },
    },
    {
      name: "legacy editor read",
      observation: observationOf("StrReplaceEditorObservation", {
        command: "view",
        path: "/workspace/legacy.ts",
      }),
      key: "OBSERVATION_MESSAGE$READ",
      values: { path: "/workspace/legacy.ts" },
    },
    {
      name: "legacy editor edit",
      observation: observationOf("StrReplaceEditorObservation", {
        command: "str_replace",
        path: "/workspace/legacy.ts",
      }),
      key: "OBSERVATION_MESSAGE$EDIT",
      values: { path: "/workspace/legacy.ts" },
    },
    {
      name: "MCP tool",
      observation: observationOf("MCPToolObservation", {
        content: [],
        is_error: false,
        tool_name: "github_search",
      }),
      key: "OBSERVATION_MESSAGE$MCP",
      values: { mcp_tool_name: "github_search" },
    },
    {
      name: "skill",
      observation: observationOf("InvokeSkillObservation", {
        skill_name: "testing",
      }),
      key: "OBSERVATION_MESSAGE$INVOKE_SKILL",
      values: { name: "testing" },
    },
    {
      name: "subagent task",
      observation: observationOf("TaskObservation", {
        subagent: "reviewer",
      }),
      key: "OBSERVATION_MESSAGE$TASK",
      values: { name: "reviewer" },
    },
    {
      name: "canvas UI",
      observation: observationOf("CanvasUIObservation", { content: [] }),
      key: "OBSERVATION_MESSAGE$CANVAS_UI",
      values: {},
    },
    {
      name: "failed model switch",
      observation: observationOf("SwitchLLMObservation", {
        is_error: true,
        profile_name: "fast",
      }),
      key: "MODEL$SWITCH_FAILED",
      values: { name: "fast" },
    },
    {
      name: "successful model switch",
      observation: observationOf("SwitchLLMObservation", {
        is_error: false,
        profile_name: "accurate",
      }),
      key: "MODEL$SWITCHED_TO_PROFILE",
      values: { name: "accurate" },
    },
    {
      name: "browser",
      observation: observationOf("BrowserObservation", {}),
      key: "OBSERVATION_MESSAGE$BROWSE",
      values: {},
    },
    {
      name: "task tracking plan",
      observation: observationOf("TaskTrackerObservation", {
        command: "plan",
        task_list: [],
      }),
      key: "OBSERVATION_MESSAGE$TASK_TRACKING_PLAN",
      values: {},
    },
    {
      name: "task tracking view",
      observation: observationOf("TaskTrackerObservation", {
        command: "view",
        task_list: [],
      }),
      key: "OBSERVATION_MESSAGE$TASK_TRACKING_VIEW",
      values: {},
    },
    {
      name: "thinking",
      observation: observationOf("ThinkObservation", { content: [] }),
      key: "OBSERVATION_MESSAGE$THINK",
      values: {},
    },
    {
      name: "glob with a long pattern",
      observation: observationOf("GlobObservation", {
        pattern: longPattern,
      }),
      key: "OBSERVATION_MESSAGE$GLOB",
      values: { pattern: `${"g".repeat(50)}...` },
    },
    {
      name: "glob without a pattern",
      observation: observationOf("GlobObservation", { pattern: "" }),
      key: "OBSERVATION_MESSAGE$GLOB",
      values: { pattern: "" },
    },
    {
      name: "grep",
      observation: observationOf("GrepObservation", { pattern: "needle" }),
      key: "OBSERVATION_MESSAGE$GREP",
      values: { pattern: "needle" },
    },
    {
      name: "grep without a pattern",
      observation: observationOf("GrepObservation", { pattern: "" }),
      key: "OBSERVATION_MESSAGE$GREP",
      values: { pattern: "" },
    },
  ])("formats the $name observation", ({ observation, key, values }) => {
    expectTranslatedTitle(createObservationEvent(observation), key, values);
  });

  it("falls back to an uppercase name for an observation without a title mapping", () => {
    const event = createObservationEvent(
      observationOf("FinishObservation", { content: [], is_error: false }),
    );

    expect(getEventContent(event).title).toBe("FINISH");
  });
});

describe("unknown and incomplete events", () => {
  it("extracts an action name from an action-like server event", () => {
    const event = {
      id: "partial-action",
      timestamp: TIMESTAMP,
      source: "agent",
      action: { kind: "SwitchLLMAction" },
    } as unknown as OpenHandsEvent;

    expect(getEventContent(event)).toEqual({
      title: "SWITCHLLM",
      details: "",
    });
  });

  it("uses the unknown translation for a normal non-tool event", () => {
    const event = createMessageEvent();

    expect(getEventContent(event)).toEqual({
      title: "translated:EVENT$UNKNOWN_EVENT",
      details: "",
    });
    expect(mocks.translate).toHaveBeenCalledWith("EVENT$UNKNOWN_EVENT");
  });

  it("does not route a non-agent action-shaped payload as an agent action", () => {
    const event = {
      id: "user-action-shape",
      timestamp: TIMESTAMP,
      source: "user",
      action: { kind: "SwitchLLMAction" },
    } as unknown as OpenHandsEvent;

    expect(getEventContent(event)).toEqual({
      title: "translated:EVENT$UNKNOWN_EVENT",
      details: "",
    });
  });

  it.each([
    ["agent event without action", {}],
    ["null action", { action: null }],
    ["primitive action", { action: "ThinkAction" }],
    ["action without kind", { action: {} }],
    ["action with non-string kind", { action: { kind: 42 } }],
    ["action with an empty kind", { action: { kind: "" } }],
  ])("safely rejects an incomplete %s payload", (_name, fields) => {
    const event = {
      id: "incomplete-action",
      timestamp: TIMESTAMP,
      source: "agent",
      ...fields,
    } as unknown as OpenHandsEvent;

    expect(getEventContent(event)).toEqual({
      title: "translated:EVENT$UNKNOWN_EVENT",
      details: "",
    });
  });
});
