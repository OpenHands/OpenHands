enum ObservationType {
  // The contents of a file
  READ = "read",

  // The diff of a file edit
  EDIT = "edit",

  // The HTML contents of a URL
  BROWSE = "browse",

  // Interactive browsing
  BROWSE_INTERACTIVE = "browse_interactive",

  // The output of a command
  RUN = "run",

  // The output of an IPython command
  RUN_IPYTHON = "run_ipython",

  // A message from the user
  CHAT = "chat",

  // Agent state has changed
  AGENT_STATE_CHANGED = "agent_state_changed",

  // Delegate result
  DELEGATE = "delegate",

  // A response to the agent's thought (usually a static message)
  THINK = "think",

  // An observation that shows agent's context extension
  RECALL = "recall",

  // A MCP tool call observation
  MCP = "mcp",

  // An error observation
  ERROR = "error",

  // A no-op observation
  NULL = "null",

  // Result of a task tracking operation
  TASK_TRACKING = "task_tracking",
}

export default ObservationType;
