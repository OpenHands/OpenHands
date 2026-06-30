/* eslint-disable i18next/no-literal-string */
import React, { useMemo, useState } from "react";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { useStopConversation } from "#/hooks/mutation/use-stop-conversation";
import { useOrgConversationStats } from "#/hooks/query/use-org-conversation-stats";
import { useOrgConversations } from "#/hooks/query/use-org-conversations";
import { useOrgUsageStats } from "#/hooks/query/use-org-usage-stats";
import { useOrgUserUsage } from "#/hooks/query/use-org-user-usage";
import { useOrganizations } from "#/hooks/query/use-organizations";
import { organizationService } from "#/api/organization-service/organization-service.api";

// Icons as inline SVGs
function SearchIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.35-4.35" />
    </svg>
  );
}

function ExportIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7,10 12,15 17,10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <circle cx="12" cy="12" r="10" />
      <rect x="9" y="9" width="6" height="6" />
    </svg>
  );
}


function TrendUpIcon() {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <polyline points="23,6 13.5,15.5 8.5,10.5 1,18" />
      <polyline points="17,6 23,6 23,12" />
    </svg>
  );
}

function TrendDownIcon() {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <polyline points="23,18 13.5,8.5 8.5,13.5 1,6" />
      <polyline points="17,18 23,18 23,12" />
    </svg>
  );
}

// Format tokens
const formatTokens = (tokens: number) => {
  if (tokens >= 1000000) return `${(tokens / 1000000).toFixed(1)}M`;
  if (tokens >= 1000) return `${(tokens / 1000).toFixed(1)}K`;
  return tokens.toString();
};

// Format cost
const formatCost = (cost: number) => {
  if (cost >= 1000) return `$${(cost / 1000).toFixed(1)}k`;
  return `$${cost.toFixed(2)}`;
};

// Format short date
const formatShortDate = (dateStr: string) => {
  const date = new Date(dateStr);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
};

// Get number of days from time window string
const getDaysFromTimeWindow = (timeWindow: string): number => {
  if (timeWindow === "ytd") return 365;
  if (timeWindow === "90d") return 90;
  if (timeWindow === "30d") return 30;
  return 7;
};

// Time window options
const TIME_WINDOWS = [
  { label: "7d", value: "7d", days: 7 },
  { label: "30d", value: "30d", days: 30 },
  { label: "90d", value: "90d", days: 90 },
  { label: "YTD", value: "ytd", days: 365 },
];

// Format date/time
const formatDateTime = (dateStr: string) => {
  const date = new Date(dateStr);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
};

const formatDateTimeOrDash = (value?: string | null) =>
  value ? formatDateTime(value) : "-";

const formatDuration = (start?: string | null, end?: string | null) => {
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

const formatAssociatedPr = (conversation: {
  pr_number?: number[];
  selected_repository?: string | null;
}) => {
  const prNumbers = conversation.pr_number ?? [];
  if (prNumbers.length === 0) return "-";
  const repo = conversation.selected_repository;
  return prNumbers
    .map((pr) => (repo ? `${repo}#${pr}` : `#${pr}`))
    .join(", ");
};

const formatBudget = (user: {
  budget_monthly_limit?: number | null;
  budget_is_disabled?: boolean;
}) => {
  if (user.budget_is_disabled) return "Disabled";
  if (user.budget_monthly_limit == null) return "-";
  return formatCost(user.budget_monthly_limit);
};

const formatMergedStatus = (merged?: boolean | null) => {
  if (merged === true) return "Yes";
  if (merged === false) return "No";
  return "-";
};



// Tabs
const TABS = ["overview", "users", "models", "conversations"] as const;
type TabType = (typeof TABS)[number];



// KPI Card component
function KPICard({
  label,
  value,
  trend,
  trendUp,
}: {
  label: string;
  value: string | number;
  trend?: string;
  trendUp?: boolean;
}) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-5">
      <span className="text-zinc-500 text-xs font-medium uppercase tracking-wide">
        {label}
      </span>
      <div className="text-white text-2xl font-bold mt-2">{value}</div>
      {trend && (
        <div
          className={`flex items-center gap-1 mt-2 text-xs ${trendUp ? "text-green-400" : "text-red-400"}`}
        >
          {trendUp ? <TrendUpIcon /> : <TrendDownIcon />}
          {trend}
        </div>
      )}
    </div>
  );
}

// Simple Area Chart component
function AreaChart({ data }: { data: { date: string; value: number }[] }) {
  const maxValue = Math.max(...data.map((d) => d.value), 1);
  const minValue = Math.min(...data.map((d) => d.value), 0);
  const range = maxValue - minValue || 1;

  // Generate SVG path
  const width = 100;
  const height = 100;
  const points = data.map((d, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((d.value - minValue) / range) * height;
    return `${x},${y}`;
  });

  // Create smooth curve path
  const pathD = `M ${points.join(" L ")}`;
  const areaD = `${pathD} L ${width},${height} L 0,${height} Z`;

  return (
    <div className="relative h-48 w-full">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-full"
        preserveAspectRatio="none"
      >
        {/* Grid lines */}
        {[0, 25, 50, 75, 100].map((pct) => (
          <line
            key={pct}
            x1="0"
            y1={`${pct}%`}
            x2="100%"
            y2={`${pct}%`}
            stroke="rgba(255,255,255,0.05)"
            strokeWidth="0.5"
          />
        ))}
        {/* Area fill */}
        <path d={areaD} fill="url(#blueGradient)" opacity="0.3" />
        {/* Line */}
        <path d={pathD} fill="none" stroke="#3B82F6" strokeWidth="2" />
        {/* Gradient definition */}
        <defs>
          <linearGradient id="blueGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#3B82F6" stopOpacity="0.5" />
            <stop offset="100%" stopColor="#3B82F6" stopOpacity="0" />
          </linearGradient>
        </defs>
      </svg>
      {/* Y-axis labels */}
      <div className="absolute left-0 top-0 bottom-0 flex flex-col justify-between text-xs text-zinc-600 -ml-2">
        <span>{maxValue.toLocaleString()}</span>
        <span>{Math.round((maxValue + minValue) / 2).toLocaleString()}</span>
        <span>{minValue.toLocaleString()}</span>
      </div>
      {/* X-axis labels */}
      <div className="absolute bottom-0 left-0 right-0 flex justify-between text-xs text-zinc-600 mt-2">
        {data
          .filter((_, i) => i % Math.ceil(data.length / 7) === 0)
          .map((d) => (
            <span key={d.date}>{formatShortDate(d.date)}</span>
          ))}
      </div>
    </div>
  );
}

export function UsageDashboard() {
  const [activeTab, setActiveTab] = useState<TabType>("overview");
  const [timeWindow, setTimeWindow] = useState("30d");
  const [modelSearch, setModelSearch] = useState("");
  const [conversationSearch, setConversationSearch] = useState("");
  const [conversationStatus, setConversationStatus] = useState("running");
  const [conversationPage, setConversationPage] = useState(1);
  const [conversationPerPage, setConversationPerPage] = useState(20);

  const { organizationId } = useSelectedOrganizationId();
  const { data: orgData } = useOrganizations();

  const days = getDaysFromTimeWindow(timeWindow);
  const { data: stats } = useOrgConversationStats();
  const { data: usageStats } = useOrgUsageStats({ days });
  const { data: userUsage, isLoading: userUsageLoading } = useOrgUserUsage();

  const conversationTimeWindow = timeWindow === "ytd" ? "" : timeWindow;

  const { data: conversationsData, isLoading: conversationsLoading } =
    useOrgConversations({
      page: conversationPage,
      perPage: conversationPerPage,
      search: conversationSearch,
      executionStatus: conversationStatus,
      timeWindow: conversationTimeWindow,
    });

  const [stoppingIds, setStoppingIds] = useState<Set<string>>(new Set());
  const [pendingStop, setPendingStop] = useState<{
    id: string;
    title: string | null;
  } | null>(null);
  const stopConversation = useStopConversation();

  const handleStop = (conversation: { id: string; title: string | null }) => {
    setPendingStop(conversation);
  };

  const confirmStop = () => {
    if (!pendingStop) return;
    const conversation = pendingStop;
    setPendingStop(null);
    setStoppingIds((prev) => {
      const next = new Set(prev);
      next.add(conversation.id);
      return next;
    });
    stopConversation.mutate(
      { conversationId: conversation.id },
      {
        onSettled: () => {
          setStoppingIds((prev) => {
            if (!prev.has(conversation.id)) return prev;
            const next = new Set(prev);
            next.delete(conversation.id);
            return next;
          });
        },
      },
    );
  };

  const cancelStop = () => {
    setPendingStop(null);
  };


  const currentOrg = orgData?.organizations?.find(
    (org) => org.id === organizationId,
  );

  const totalConversations = usageStats?.agent_runs ?? 0;
  const activeConversations = stats?.active_conversations ?? 0;
  const avgCostPerConversation =
    totalConversations > 0
      ? (usageStats?.estimated_spend ?? 0) / totalConversations
      : 0;
  const totalSpend = formatCost(usageStats?.estimated_spend ?? 0);

  const modelRows = useMemo(() => {
    return (usageStats?.model_usage ?? []).map((model) => {
      const avgTokens =
        model.conversation_count > 0
          ? Math.round(model.total_tokens / model.conversation_count)
          : 0;
      const avgCost =
        model.conversation_count > 0
          ? model.total_cost / model.conversation_count
          : 0;
      return {
        ...model,
        avgTokens,
        avgCost,
      };
    });
  }, [usageStats?.model_usage]);

  const filteredModels = useMemo(() => {
    return modelRows.filter((model) =>
      model.model_name.toLowerCase().includes(modelSearch.toLowerCase()),
    );
  }, [modelRows, modelSearch]);

  const chartData = useMemo(
    () =>
      (usageStats?.daily_usage ?? []).map((d) => ({
        date: d.date,
        value: d.conversations,
      })),
    [usageStats?.daily_usage],
  );

  const tabCounts = {
    overview: null,
    users: userUsage?.items.length ?? 0,
    models: modelRows.length,
    conversations: conversationsData?.total_items ?? 0,
  };

  const timeWindowLabel = timeWindow === "ytd" ? "YTD" : timeWindow.toUpperCase();

  const conversationTotalPages = conversationsData?.total_pages ?? 1;
  const conversationTotalItems = conversationsData?.total_items ?? 0;

  const pendingStopLabel = pendingStop?.title?.trim();
  const stopConfirmationText = pendingStopLabel
    ? `Stop "${pendingStopLabel}"? This will cancel any in-progress agent run.`
    : "Stop this conversation? This will cancel any in-progress agent run.";




  const exportUrl = useMemo(() => {
    if (!organizationId) return "#";
    return organizationService.exportConversationsUrl({
      orgId: organizationId,
      search: conversationSearch || undefined,
      executionStatus: conversationStatus || undefined,
      timeWindow: conversationTimeWindow || undefined,
    });
  }, [
    organizationId,
    conversationSearch,
    conversationStatus,
    conversationTimeWindow,
  ]);

  return (
    <div className="min-h-screen bg-zinc-950 text-white">
      {/* Header */}
      <div className="px-8 py-6 border-b border-zinc-800">
        <div className="flex items-start justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-white mb-1">
              Usage & Monitoring
            </h1>
            <p className="text-zinc-400">
              Monitor adoption, spend, and ROI across{" "}
              {currentOrg?.name || "your organization"}.
            </p>
          </div>
          {/* Time window selector */}
          <div className="flex items-center gap-1 bg-zinc-900 border border-zinc-800 rounded-lg p-1">
            {TIME_WINDOWS.map((tw) => (
              <button
                key={tw.value}
                type="button"
                onClick={() => {
                  setTimeWindow(tw.value);
                  setConversationPage(1);
                }}
                className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
                  timeWindow === tw.value
                    ? "bg-zinc-800 text-white"
                    : "text-zinc-400 hover:text-white"
                }`}
              >
                {tw.label}
              </button>
            ))}
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-6 border-b border-zinc-800 -mb-6">
          {TABS.map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => setActiveTab(tab)}
              className={`flex items-center gap-2 px-1 py-3 text-sm font-medium transition-colors border-b-2 ${
                activeTab === tab
                  ? "border-blue-500 text-white"
                  : "border-transparent text-zinc-400 hover:text-white"
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
              {typeof tabCounts[tab] === "number" && (
                <span className="px-2 py-0.5 text-xs bg-zinc-800 text-zinc-400 rounded-full">
                  {tabCounts[tab].toLocaleString()}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="p-8">
        {/* Overview Tab */}
        {activeTab === "overview" && (
          <div className="space-y-6">
            {/* KPI Row */}
            <div className="grid grid-cols-4 gap-4">
              <KPICard
                label="Total Conversations"
                value={totalConversations.toLocaleString()}
              />
              <KPICard
                label="Active Conversations"
                value={activeConversations.toLocaleString()}
              />
              <KPICard
                label="Avg Cost / Conversation"
                value={`$${avgCostPerConversation.toFixed(2)}`}
              />
              <KPICard
                label={`Total Spend (${timeWindowLabel})`}
                value={totalSpend}
              />
            </div>

            {/* Chart Row */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6">
              <div className="flex items-start justify-between mb-6">
                <div>
                  <h2 className="text-lg font-medium text-white">
                    Conversations per day
                  </h2>
                  <p className="text-sm text-zinc-500">
                    {timeWindowLabel} · all users
                  </p>
                </div>
                <button
                  type="button"
                  className="flex items-center gap-2 px-3 py-1.5 text-sm text-zinc-400 border border-zinc-700 rounded-lg hover:text-white hover:border-zinc-600 transition-colors"
                >
                  <ExportIcon />
                  Export CSV
                </button>
              </div>
              {chartData.length > 0 ? (
                <AreaChart data={chartData} />
              ) : (
                <div className="py-10 text-center text-sm text-zinc-500">
                  No usage data available yet.
                </div>
              )}
            </div>

          </div>
        )}

        {/* Conversations Tab */}
        {activeTab === "conversations" && (
          <div className="space-y-4">
            {/* Filter Bar */}
            <div className="flex items-center gap-4">
              <div className="relative flex-1 max-w-md">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500">
                  <SearchIcon />
                </span>
                <input
                  type="text"
                  placeholder="Search by title or user..."
                  value={conversationSearch}
                  onChange={(e) => {
                    setConversationSearch(e.target.value);
                    setConversationPage(1);
                  }}
                  className="w-full pl-10 pr-4 py-2 bg-zinc-900 border border-zinc-800 rounded-lg text-white placeholder-zinc-500 focus:outline-none focus:border-zinc-700"
                />
              </div>
              <select
                value={conversationStatus}
                onChange={(e) => {
                  setConversationStatus(e.target.value);
                  setConversationPage(1);
                }}
                className="px-3 py-2 bg-zinc-900 border border-zinc-800 rounded-lg text-sm text-zinc-400 focus:outline-none focus:border-zinc-700"
              >
                <option value="">All statuses</option>
                <option value="running">Running</option>
                <option value="idle">Idle</option>
                <option value="paused">Paused</option>
                <option value="finished">Finished</option>
                <option value="error">Error</option>
                <option value="stuck">Stuck</option>
              </select>
              <a
                href={exportUrl}
                className="flex items-center gap-2 px-3 py-2 text-sm text-zinc-400 border border-zinc-700 rounded-lg hover:text-white hover:border-zinc-600 transition-colors"
              >
                <ExportIcon />
                Export CSV
              </a>
            </div>

            {/* Conversations Table */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-zinc-800">
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      User
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Tokens
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Spend
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Duration
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Started
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Last update
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Associated PR
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Merged?
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Stop
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {conversationsLoading && (
                    <tr>
                      <td
                        colSpan={10}
                        className="px-4 py-8 text-center text-zinc-500"
                      >
                        Loading conversations...
                      </td>
                    </tr>
                  )}
                  {!conversationsLoading &&
                    (conversationsData?.items.length ?? 0) === 0 && (
                      <tr>
                        <td
                          colSpan={10}
                          className="px-4 py-8 text-center text-zinc-500"
                        >
                          No conversations found for this time window.
                        </td>
                      </tr>
                    )}
                  {conversationsData?.items.map((conversation) => {
                    const isRunning =
                      conversation.execution_status?.toLowerCase() === "running";
                    return (
                      <tr
                        key={conversation.id}
                        className="border-b border-zinc-800/50 hover:bg-zinc-800/50 transition-colors"
                      >
                        <td className="px-4 py-4">
                          <div className="text-white text-sm font-medium">
                            {conversation.user_email?.split("@")[0] || "Unknown"}
                          </div>
                          <div className="text-xs text-zinc-500">
                            {conversation.user_email || "-"}
                          </div>
                        </td>
                        <td className="px-4 py-4 text-right text-sm font-mono text-white">
                          {formatTokens(conversation.total_tokens)}
                        </td>
                        <td className="px-4 py-4 text-right text-sm text-white">
                          {formatCost(conversation.accumulated_cost)}
                        </td>
                        <td className="px-4 py-4 text-sm text-zinc-400">
                          {formatDuration(
                            conversation.created_at,
                            conversation.updated_at,
                          )}
                        </td>
                        <td className="px-4 py-4 text-sm text-zinc-400">
                          {formatDateTimeOrDash(conversation.created_at)}
                        </td>
                        <td className="px-4 py-4 text-sm text-zinc-400">
                          {formatDateTimeOrDash(conversation.updated_at)}
                        </td>
                        <td className="px-4 py-4 text-sm text-zinc-400">
                          {formatAssociatedPr(conversation)}
                        </td>
                        <td className="px-4 py-4 text-sm text-zinc-400">
                          {formatMergedStatus(conversation.pr_merged)}
                        </td>
                        <td className="px-4 py-4 text-sm text-zinc-400 capitalize">
                          {conversation.trigger || conversation.agent_kind || "-"}
                        </td>
                        <td className="px-4 py-4 text-right text-sm">
                          {isRunning && (
                            <button
                              type="button"
                              onClick={() =>
                                handleStop({
                                  id: conversation.id,
                                  title: conversation.title ?? null,
                                })
                              }
                              disabled={stoppingIds.has(conversation.id)}
                              className="inline-flex items-center gap-1.5 px-2 py-1 text-xs text-zinc-400 hover:text-white hover:bg-zinc-800 rounded transition-colors disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-zinc-400 disabled:cursor-not-allowed"
                              title="Stop conversation"
                              aria-label="Stop conversation"
                            >
                              <StopIcon />
                              {stoppingIds.has(conversation.id)
                                ? "Stopping…"
                                : "Stop"}
                            </button>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <div className="flex items-center justify-between px-4 py-3 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() =>
                      setConversationPage((prev) => Math.max(1, prev - 1))
                    }
                    disabled={conversationPage <= 1}
                    className={`flex items-center gap-1 px-2 py-1 text-sm rounded transition-colors ${
                      conversationPage <= 1
                        ? "text-zinc-600 cursor-not-allowed"
                        : "text-zinc-400 hover:text-white hover:bg-zinc-800"
                    }`}
                  >
                    Previous
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      setConversationPage((prev) =>
                        Math.min(conversationTotalPages, prev + 1),
                      )
                    }
                    disabled={conversationPage >= conversationTotalPages}
                    className={`flex items-center gap-1 px-2 py-1 text-sm rounded transition-colors ${
                      conversationPage >= conversationTotalPages
                        ? "text-zinc-600 cursor-not-allowed"
                        : "text-zinc-400 hover:text-white hover:bg-zinc-800"
                    }`}
                  >
                    Next
                  </button>
                </div>
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <span className="text-zinc-500 text-sm">Per page</span>
                    <select
                      value={conversationPerPage}
                      onChange={(e) => {
                        setConversationPerPage(Number(e.target.value));
                        setConversationPage(1);
                      }}
                      className="px-2 py-1 bg-zinc-900 border border-zinc-800 rounded text-sm text-white focus:outline-none"
                    >
                      <option value="10">10</option>
                      <option value="20">20</option>
                      <option value="50">50</option>
                    </select>
                  </div>
                  <span className="text-zinc-500 text-sm">
                    Page {conversationPage} of {conversationTotalPages} · {conversationTotalItems} conversations
                  </span>
                </div>
              </div>
            </div>

            {pendingStop && (
              <ConfirmationModal
                text={stopConfirmationText}
                onConfirm={confirmStop}
                onCancel={cancelStop}
              />
            )}

          </div>
        )}

        {/* Users Tab */}
        {activeTab === "users" && (
          <div className="space-y-4">
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-zinc-800">
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      User
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Convos
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      First convo
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Last convo
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      First login
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Last login
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Spend MTD
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Spend YTD
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Lifetime
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Budget
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      PRs merged
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {userUsageLoading && (
                    <tr>
                      <td
                        colSpan={11}
                        className="px-4 py-8 text-center text-zinc-500"
                      >
                        Loading user usage...
                      </td>
                    </tr>
                  )}
                  {!userUsageLoading && (userUsage?.items.length ?? 0) === 0 && (
                    <tr>
                      <td
                        colSpan={11}
                        className="px-4 py-8 text-center text-zinc-500"
                      >
                        No user usage data available yet.
                      </td>
                    </tr>
                  )}
                  {userUsage?.items.map((user) => (
                    <tr
                      key={user.user_id}
                      className="border-b border-zinc-800/50 hover:bg-zinc-800/50 transition-colors"
                    >
                      <td className="px-4 py-4">
                        <div className="text-white text-sm font-medium">
                          {user.user_name ??
                            user.user_email?.split("@")[0] ??
                            "Unknown"}
                        </div>
                        <div className="text-xs text-zinc-500">
                          {user.user_email || "-"}
                        </div>
                      </td>
                      <td className="px-4 py-4 text-right text-sm text-white">
                        {user.conversation_count.toLocaleString()}
                      </td>
                      <td className="px-4 py-4 text-sm text-zinc-400">
                        {formatDateTimeOrDash(user.first_conversation_at)}
                      </td>
                      <td className="px-4 py-4 text-sm text-zinc-400">
                        {formatDateTimeOrDash(user.last_conversation_at)}
                      </td>
                      <td className="px-4 py-4 text-sm text-zinc-400">
                        {formatDateTimeOrDash(user.first_login_at)}
                      </td>
                      <td className="px-4 py-4 text-sm text-zinc-400">
                        {formatDateTimeOrDash(user.last_login_at)}
                      </td>
                      <td className="px-4 py-4 text-right text-sm text-white">
                        {formatCost(user.spend_mtd)}
                      </td>
                      <td className="px-4 py-4 text-right text-sm text-white">
                        {formatCost(user.spend_ytd)}
                      </td>
                      <td className="px-4 py-4 text-right text-sm text-white">
                        {formatCost(user.spend_lifetime)}
                      </td>
                      <td className="px-4 py-4 text-sm text-zinc-400">
                        {formatBudget(user)}
                      </td>
                      <td className="px-4 py-4 text-right text-sm text-zinc-400">
                        {user.prs_merged ?? "-"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Models Tab */}
        {activeTab === "models" && (
          <div className="space-y-4">
            {/* Filter Bar */}
            <div className="flex items-center justify-between">
              <div className="relative w-64">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500">
                  <SearchIcon />
                </span>
                <input
                  type="text"
                  placeholder="Search models..."
                  value={modelSearch}
                  onChange={(e) => setModelSearch(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-zinc-900 border border-zinc-800 rounded-lg text-white placeholder-zinc-500 focus:outline-none focus:border-zinc-700"
                />
              </div>
              <button
                type="button"
                className="flex items-center gap-2 px-3 py-2 text-sm text-zinc-400 border border-zinc-700 rounded-lg hover:text-white hover:border-zinc-600 transition-colors"
              >
                <ExportIcon />
                Export CSV
              </button>
            </div>

            {/* Models Table */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-zinc-800">
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Model
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Conversations
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Tokens Used
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Avg Tokens / Convo
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Avg Cost / Convo
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      Total Cost
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {filteredModels.map((model) => (
                    <tr
                      key={model.model_name}
                      className="border-b border-zinc-800/50 hover:bg-zinc-800/50 transition-colors"
                    >
                      <td className="px-4 py-5">
                        <div className="text-white font-medium">
                          {model.model_name}
                        </div>
                      </td>
                      <td className="px-4 py-5 text-white text-sm font-mono text-right">
                        {model.conversation_count.toLocaleString()}
                      </td>
                      <td className="px-4 py-5 text-white text-sm font-mono text-right">
                        {formatTokens(model.total_tokens)}
                      </td>
                      <td className="px-4 py-5 text-white text-sm font-mono text-right">
                        {formatTokens(model.avgTokens)}
                      </td>
                      <td className="px-4 py-5 text-white text-sm font-mono text-right">
                        ${model.avgCost.toFixed(2)}
                      </td>
                      <td className="px-4 py-5 text-white text-sm font-mono text-right font-medium">
                        ${model.total_cost.toFixed(2)}
                      </td>
                    </tr>
                  ))}

                  {filteredModels.length === 0 && (
                    <tr>
                      <td
                        colSpan={6}
                        className="px-4 py-8 text-center text-zinc-500"
                      >
                        No model usage data available for this time window.
                      </td>
                    </tr>
                  )}

                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
