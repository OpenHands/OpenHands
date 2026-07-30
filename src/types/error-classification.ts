export interface ErrorClassification {
  origin:
    | "sdk"
    | "provider"
    | "agent"
    | "tool"
    | "mcp"
    | "environment"
    | "unknown";
  cause: string;
  blame:
    | "user_configuration"
    | "external"
    | "agent_behavior"
    | "product_defect"
    | "unknown";
  impact: "notice" | "step_failed" | "run_stopped" | "conversation_unusable";
  retry: "none" | "immediate" | "after_backoff" | "after_user_action";
  user_action:
    | "none"
    | "retry"
    | "reauthenticate"
    | "configure_llm"
    | "select_model"
    | "contact_support";
  presentation: "info" | "warning" | "error";
  telemetry: "none" | "outcome" | "diagnostic";
}
