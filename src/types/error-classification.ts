export interface ErrorClassification {
  kind:
    | "auth"
    | "quota"
    | "rate_limit"
    | "config"
    | "transient"
    | "agent_action"
    | "internal"
    | "unknown";
  retryable: boolean;
  user_action: "none" | "retry" | "settings";
  error_id?: string | null;
}
