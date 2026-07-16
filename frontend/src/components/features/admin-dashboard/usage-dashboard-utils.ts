/* eslint-disable i18next/no-literal-string */

export const TIME_WINDOWS = [
  { label: "7d", value: "7d" },
  { label: "30d", value: "30d" },
  { label: "90d", value: "90d" },
  { label: "YTD", value: "ytd" },
];

export const AGENT_COLORS = [
  "#3B82F6",
  "#F59E0B",
  "#10B981",
  "#8B5CF6",
  "#EF4444",
  "#06B6D4",
  "#F97316",
];

export const formatTokens = (tokens: number) => {
  if (tokens >= 1000000) return `${(tokens / 1000000).toFixed(1)}M`;
  if (tokens >= 1000) return `${(tokens / 1000).toFixed(1)}K`;
  return tokens.toString();
};

export const formatCost = (cost: number) => {
  if (cost >= 1000) return `$${(cost / 1000).toFixed(1)}k`;
  return `$${cost.toFixed(2)}`;
};

export const formatShortDate = (dateStr: string) => {
  const date = new Date(dateStr);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
};

export const formatDateTime = (dateStr: string) => {
  const date = new Date(dateStr);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
};

export const formatDateTimeOrDash = (value?: string | null) =>
  value ? formatDateTime(value) : "-";

export const formatDuration = (start?: string | null, end?: string | null) => {
  if (!start || !end) return "-";
  const startMs = new Date(start).getTime();
  const endMs = new Date(end).getTime();
  if (Number.isNaN(startMs) || Number.isNaN(endMs)) return "-";
  const diffMs = Math.max(0, endMs - startMs);
  const totalMinutes = Math.floor(diffMs / 60000);
  const totalHours = Math.floor(totalMinutes / 60);
  if (totalHours >= 24) {
    const days = Math.floor(totalHours / 24);
    const hours = totalHours % 24;
    return `${days}d ${hours}h`;
  }
  if (totalHours > 0) {
    return `${totalHours}h ${totalMinutes % 60}m`;
  }
  return `${totalMinutes}m`;
};

export const formatAssociatedPr = (conversation: {
  pr_number?: number[];
  selected_repository?: string | null;
}) => {
  const prNumbers = conversation.pr_number ?? [];
  if (prNumbers.length === 0) return "-";
  const repo = conversation.selected_repository;
  return prNumbers.map((pr) => (repo ? `${repo}#${pr}` : `#${pr}`)).join(", ");
};

export const formatBudget = (user: {
  budget_monthly_limit?: number | null;
  budget_is_disabled?: boolean;
}) => {
  if (user.budget_is_disabled) return "Disabled";
  if (user.budget_monthly_limit == null) return "-";
  return formatCost(user.budget_monthly_limit);
};

export const formatAgentLabel = (conversation: {
  agent_kind?: string | null;
  llm_model?: string | null;
}) => {
  const agentKind = conversation.agent_kind ?? null;
  if (agentKind === "acp") {
    const llmModel = conversation.llm_model ?? "";
    const llmModelLower = llmModel.toLowerCase();
    if (!llmModel) return "ACP";
    if (llmModelLower.includes("claude")) return "Claude";
    if (llmModelLower.includes("codex")) return "Codex";
    if (llmModelLower.includes("gpt") || llmModelLower.includes("openai")) {
      return "OpenAI";
    }
    if (llmModelLower.includes("gemini")) return "Gemini";
    return llmModel;
  }
  if (agentKind === "openhands") return "OpenHands";
  return agentKind || "-";
};

export const formatMergedStatus = (merged?: boolean | null) => {
  if (merged === true) return "Yes";
  if (merged === false) return "No";
  return "-";
};
