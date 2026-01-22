export enum AgentState {
  LOADING = "loading",
  INIT = "init",
  RUNNING = "running",
  AWAITING_USER_INPUT = "awaiting_user_input",
  PAUSED = "paused",
  STOPPED = "stopped",
  FINISHED = "finished",
  REJECTED = "rejected",
  ERROR = "error",
  RATE_LIMITED = "rate_limited",
  AWAITING_USER_CONFIRMATION = "awaiting_user_confirmation",
  USER_CONFIRMED = "user_confirmed",
  USER_REJECTED = "user_rejected",
  DELETE = "delete",
}

export const RUNTIME_INACTIVE_STATES = [
  AgentState.INIT,
  AgentState.LOADING,
  // Removed AgentState.STOPPED to allow tabs to remain visible when agent is stopped
  AgentState.ERROR,
];
