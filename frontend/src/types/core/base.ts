export type OpenHandsEventType =
  | "message"
  | "system"
  | "agent_state_changed"
  | "change_agent_state"
  | "run"
  | "read"
  | "write"
  | "edit"
  | "run_ipython"
  | "delegate"
  | "browse"
  | "browse_interactive"
  | "reject"
  | "think"
  | "finish"
  | "error"
  | "recall"
  | "mcp"
  | "call_tool_mcp"
  | "task_tracking"
  | "user_rejected";

export type OpenHandsSourceType = "agent" | "user" | "environment";

interface OpenHandsBaseEvent {
  id: number;
  source: OpenHandsSourceType;
  message: string;
  timestamp: string; // ISO 8601
}

export interface OpenHandsActionEvent<
  T extends OpenHandsEventType,
> extends OpenHandsBaseEvent {
  action: T;
  args: Record<string, unknown>;
}

export interface OpenHandsObservationEvent<
  T extends OpenHandsEventType,
> extends OpenHandsBaseEvent {
  cause: number;
  observation: T;
  content: string;
  extras: Record<string, unknown>;
}
