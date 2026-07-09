/* eslint-disable i18next/no-literal-string */
import React, { useMemo, useState } from "react";
import { useSearchParams } from "react-router";
import toast from "react-hot-toast";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { useOrganizations } from "#/hooks/query/use-organizations";
import { useOrgConversationStats } from "#/hooks/query/use-org-conversation-stats";
import { useOrgConversations } from "#/hooks/query/use-org-conversations";
import { useOrgUsageStats } from "#/hooks/query/use-org-usage-stats";
import { useStopConversation } from "#/hooks/mutation/use-stop-conversation";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { organizationService } from "#/api/organization-service/organization-service.api";

// Conversation statuses from which the user can no longer stop a running
// agent. Mirrors the terminal-status set used by OrgConversationService.
const TERMINAL_EXECUTION_STATUSES = new Set([
  "finished",
  "error",
  "stuck",
  "deleting",
]);

const isStoppable = (status: string | null | undefined) =>
  !TERMINAL_EXECUTION_STATUSES.has((status ?? "").toLowerCase());

// Icons as inline SVGs for simplicity
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

function CopyIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function ExternalLinkIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <polyline points="15,3 21,3 21,9" />
      <line x1="10" y1="14" x2="21" y2="3" />
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

function ChevronLeftIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <polyline points="15,18 9,12 15,6" />
    </svg>
  );
}

function ChevronRightIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <polyline points="9,18 15,12 9,6" />
    </svg>
  );
}

function ActivityIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <polyline points="22,12 18,12 15,21 9,3 6,12 2,12" />
    </svg>
  );
}

function ServerIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <rect x="2" y="2" width="20" height="8" rx="2" ry="2" />
      <rect x="2" y="14" width="20" height="8" rx="2" ry="2" />
      <line x1="6" y1="6" x2="6.01" y2="6" />
      <line x1="6" y1="18" x2="6.01" y2="18" />
    </svg>
  );
}

function CheckCircleIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <polyline points="22,4 12,14.01 9,11.01" />
    </svg>
  );
}

function DollarIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <line x1="12" y1="1" x2="12" y2="23" />
      <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
    </svg>
  );
}

// Status badge component
function StatusBadge({ status }: { status: string | null }) {
  const getStatusStyle = () => {
    switch (status) {
      case "running":
        return "bg-blue-500/10 text-blue-400 border-blue-500/30";
      case "idle":
      case "paused":
        return "bg-yellow-500/10 text-yellow-400 border-yellow-500/30";
      case "finished":
        return "bg-green-500/10 text-green-400 border-green-500/30";
      case "error":
      case "stuck":
        return "bg-red-500/10 text-red-400 border-red-500/30";
      default:
        return "bg-gray-500/10 text-gray-400 border-gray-500/30";
    }
  };

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${getStatusStyle()}`}
    >
      {status || "unknown"}
    </span>
  );
}

// Runtime status badge
function RuntimeBadge({ status }: { status: string | null }) {
  const getStatusStyle = () => {
    switch (status) {
      case "RUNNING":
        return "bg-blue-500/10 text-blue-400 border-blue-500/30";
      case "STARTING":
        return "bg-yellow-500/10 text-yellow-400 border-yellow-500/30";
      case "ERROR":
      case "MISSING":
        return "bg-red-500/10 text-red-400 border-red-500/30";
      case "PAUSED":
        return "bg-gray-500/10 text-gray-400 border-gray-500/30";
      default:
        return "bg-gray-500/10 text-gray-400 border-gray-500/30";
    }
  };

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${getStatusStyle()}`}
    >
      {status || "N/A"}
    </span>
  );
}

// KPI Card component
function KPICard({
  label,
  value,
  subtext,
  icon,
}: {
  label: string;
  value: string | number;
  subtext?: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="bg-[#1E1E1E] border border-[#262626] rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-[#8C8C8C] text-xs font-medium uppercase tracking-wide">
          {label}
        </span>
        <span className="text-[#8C8C8C]">{icon}</span>
      </div>
      <div className="text-white text-2xl font-bold mb-1">{value}</div>
      {subtext && <div className="text-[#8C8C8C] text-xs">{subtext}</div>}
    </div>
  );
}

// Format tokens
const formatTokens = (tokens: number) => {
  if (tokens >= 1000000) return `${(tokens / 1000000).toFixed(1)}M`;
  if (tokens >= 1000) return `${(tokens / 1000).toFixed(1)}k`;
  return tokens.toString();
};

// Format cost
const formatCost = (cost: number) => {
  if (cost >= 1000) return `$${(cost / 1000).toFixed(1)}k`;
  return `$${cost.toFixed(2)}`;
};

// Format date
const formatDate = (dateStr: string) => {
  const date = new Date(dateStr);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
};

// Copy to clipboard
const copyToClipboard = (text: string) => {
  navigator.clipboard.writeText(text);
};

// Get number of days from time window string
const getDaysFromTimeWindow = (timeWindow: string): number => {
  if (timeWindow === "90d") return 90;
  if (timeWindow === "30d") return 30;
  return 7;
};

export function AdminDashboard() {
  const { organizationId } = useSelectedOrganizationId();
  const { data: orgData } = useOrganizations();
  const [activeTab, setActiveTab] = useState<"conversations" | "usage">(
    "conversations",
  );
  const [searchParams, setSearchParams] = useSearchParams();

  // Use the currently selected org (from the org switcher) rather than
  // deriving from the org list — avoids mismatches when the user belongs
  // to multiple orgs and has different roles in each.
  const orgId = organizationId ?? undefined;
  const currentOrg = orgData?.organizations?.find(
    (org) => org.id === organizationId,
  );

  // Filter state from URL params
  const page = parseInt(searchParams.get("page") || "1", 10);
  const perPage = parseInt(searchParams.get("per_page") || "20", 10);
  const search = searchParams.get("search") || "";
  const sortBy = searchParams.get("sort_by") || "updated_at";
  const sortOrder = searchParams.get("sort_order") || "desc";
  const executionStatus = searchParams.get("execution_status") || "";
  const sandboxStatus = searchParams.get("sandbox_status") || "";
  const timeWindow = searchParams.get("time_window") || "";

  // Fetch stats
  const { data: stats } = useOrgConversationStats();

  // Fetch usage stats for Usage tab
  const { data: usageStats } = useOrgUsageStats({
    days: getDaysFromTimeWindow(timeWindow),
  });

  // Fetch conversations
  const { data: conversationsData, isLoading: conversationsLoading } =
    useOrgConversations({
      page,
      perPage,
      search,
      sortBy,
      sortOrder,
      executionStatus,
      sandboxStatus,
      timeWindow,
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
    const toastId = toast.loading("Stopping conversation…");
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
          toast.dismiss(toastId);
        },
      },
    );
  };

  const cancelStop = () => {
    setPendingStop(null);
  };

  const updateFilter = (key: string, value: string | string[] | null) => {
    const newParams = new URLSearchParams(searchParams);
    if (value === null || value === "") {
      newParams.delete(key);
    } else if (Array.isArray(value)) {
      newParams.delete(key);
      value.forEach((v) => newParams.append(key, v));
    } else {
      newParams.set(key, value);
    }
    newParams.set("page", "1"); // Reset to first page on filter change
    setSearchParams(newParams);
  };

  const handlePageChange = (newPage: number) => {
    const newParams = new URLSearchParams(searchParams);
    newParams.set("page", String(newPage));
    setSearchParams(newParams);
  };

  const totalPages = conversationsData?.total_pages || 1;
  const totalItems = conversationsData?.total_items || 0;
  const pendingStopLabel = pendingStop?.title?.trim();
  const stopConfirmationText = pendingStopLabel
    ? `Stop "${pendingStopLabel}"? This will cancel any in-progress agent run.`
    : "Stop this conversation? This will cancel any in-progress agent run.";

  const exportUrl = useMemo(() => {
    if (!orgId) return "#";
    return organizationService.exportConversationsUrl({
      orgId,
      search: search || undefined,
      sortBy,
      sortOrder,
      executionStatus: executionStatus.length ? executionStatus : undefined,
      sandboxStatus: sandboxStatus.length ? sandboxStatus : undefined,
      timeWindow: timeWindow || undefined,
    });
  }, [
    orgId,
    search,
    sortBy,
    sortOrder,
    executionStatus,
    sandboxStatus,
    timeWindow,
  ]);

  if (!orgId) {
    return (
      <div className="p-8 text-center text-[#8C8C8C]">
        Please select an organization to view usage & monitoring.
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0B0B0B] text-white flex flex-col">
      {/* Top Header with Tabs */}
      <header className="border-b border-white/10 px-6 py-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-white font-bold text-lg">Admin dashboard</h1>
            <p className="text-[#888888] text-sm mt-1">
              {currentOrg?.name || "your organization"}
            </p>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 border-b border-white/10 -mb-4">
          <button
            type="button"
            onClick={() => setActiveTab("usage")}
            className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 ${
              activeTab === "usage"
                ? "border-white text-white"
                : "border-transparent text-[#888888] hover:text-white"
            }`}
          >
            Usage
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("conversations")}
            className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 ${
              activeTab === "conversations"
                ? "border-white text-white"
                : "border-transparent text-[#888888] hover:text-white"
            }`}
          >
            Conversations
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        {activeTab === "usage" ? (
          <>
            {/* Usage Page Header */}
            <div className="p-8 border-b border-white/10">
              <div className="flex items-start justify-between mb-2">
                <div>
                  <p className="text-[#888888] text-sm">
                    At-a-glance OpenHands usage for{" "}
                    {currentOrg?.name || "your organization"}.
                  </p>
                </div>
                <button
                  type="button"
                  className="flex items-center gap-2 px-4 py-2 rounded-full border border-white/20 text-sm text-[#888888] hover:text-white hover:border-white/40 transition-colors"
                >
                  Last 7 days
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <path d="M6 9l6 6 6-6" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Usage Content */}
            <div className="p-8">
              {/* Top Metrics Row - 3 cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                {/* Active Users */}
                <div className="bg-[#1A1A1A] p-6 rounded-xl border border-white/5">
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-[#888888] text-xs font-medium uppercase tracking-wide">
                      Active Users
                    </span>
                    <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center">
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                      >
                        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                        <circle cx="9" cy="7" r="4" />
                        <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                        <path d="M16 3.13a4 4 0 0 1 0 7.75" />
                      </svg>
                    </div>
                  </div>
                  <div className="text-white text-3xl font-bold tracking-tight mb-2">
                    {usageStats?.active_users ?? 0}
                  </div>
                  <div className="text-[#888888] text-sm">
                    users in last 7 days
                  </div>
                </div>

                {/* Agent Runs */}
                <div className="bg-[#1A1A1A] p-6 rounded-xl border border-white/5">
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-[#888888] text-xs font-medium uppercase tracking-wide">
                      Agent Runs
                    </span>
                    <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center">
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                      >
                        <polyline points="22,12 18,12 15,21 9,3 6,12 2,12" />
                      </svg>
                    </div>
                  </div>
                  <div className="text-white text-3xl font-bold tracking-tight mb-2">
                    {usageStats?.agent_runs ?? 0}
                  </div>
                  <div className="text-[#888888] text-sm">
                    {(() => {
                      const days = getDaysFromTimeWindow(timeWindow);
                      if (usageStats?.agent_runs && usageStats.agent_runs > 0) {
                        return `${Math.round(usageStats.agent_runs / days)} daily average`;
                      }
                      return `No runs in ${days} days`;
                    })()}
                  </div>
                </div>

                {/* Token Usage */}
                <div className="bg-[#1A1A1A] p-6 rounded-xl border border-white/5">
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-[#888888] text-xs font-medium uppercase tracking-wide">
                      Token Usage
                    </span>
                    <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center">
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                      >
                        <circle cx="12" cy="12" r="10" />
                        <path d="M12 6v6l4 2" />
                      </svg>
                    </div>
                  </div>
                  <div className="text-white text-3xl font-bold tracking-tight mb-2">
                    {formatTokens(usageStats?.total_tokens ?? 0)}
                  </div>
                  <div className="text-[#888888] text-sm">
                    ${(usageStats?.estimated_spend ?? 0).toFixed(2)} estimated
                    spend
                  </div>
                </div>
              </div>

              {/* Bottom Row - Chart and Team Usage */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {/* Daily Usage Chart */}
                <div className="lg:col-span-2 bg-[#1A1A1A] p-6 rounded-xl border border-white/5">
                  <div className="mb-6">
                    <h2 className="text-white font-medium text-lg">
                      Daily usage
                    </h2>
                    <p className="text-[#888888] text-sm">
                      Tokens and conversations by day
                    </p>
                  </div>

                  {/* Simple Bar Chart */}
                  <div className="h-48 flex items-end justify-between gap-4 px-2">
                    {(usageStats?.daily_usage ?? []).map((day) => {
                      // Find max tokens for scaling
                      const maxTokens = Math.max(
                        ...(usageStats?.daily_usage?.map((d) => d.tokens) ?? [
                          1,
                        ]),
                        1,
                      );
                      const tokenHeight =
                        day.tokens > 0 ? (day.tokens / maxTokens) * 100 : 5;
                      const convoHeight =
                        day.conversations > 0
                          ? Math.min(
                              (day.conversations /
                                Math.max(
                                  ...(usageStats?.daily_usage?.map(
                                    (d) => d.conversations,
                                  ) ?? [1]),
                                  1,
                                )) *
                                100,
                              80,
                            )
                          : 5;

                      // Get day name from date
                      const date = new Date(day.date);
                      const dayName = date.toLocaleDateString("en-US", {
                        weekday: "short",
                      });

                      return (
                        <div
                          key={day.date}
                          className="flex-1 flex flex-col items-center gap-2"
                        >
                          <div className="w-full flex items-end justify-center gap-1 h-36">
                            <div
                              className="w-4 bg-white rounded-sm transition-all"
                              style={{ height: `${tokenHeight}%` }}
                              title={`${day.tokens.toLocaleString()} tokens`}
                            />
                            <div
                              className="w-4 bg-zinc-600 rounded-sm transition-all"
                              style={{ height: `${convoHeight}%` }}
                              title={`${day.conversations} conversations`}
                            />
                          </div>
                          <span className="text-[#888888] text-xs">
                            {dayName}
                          </span>
                        </div>
                      );
                    })}
                  </div>

                  {/* Legend */}
                  <div className="flex items-center gap-6 mt-6 pt-4 border-t border-white/10">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-sm bg-white" />
                      <span className="text-[#888888] text-sm">Tokens</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-sm bg-zinc-600" />
                      <span className="text-[#888888] text-sm">
                        Conversations
                      </span>
                    </div>
                  </div>
                </div>

                {/* Usage by Team (User) */}
                <div className="bg-[#1A1A1A] p-6 rounded-xl border border-white/5">
                  <div className="mb-6">
                    <h2 className="text-white font-medium text-lg">
                      Usage by user
                    </h2>
                    <p className="text-[#888888] text-sm">
                      Share of total runs
                    </p>
                  </div>

                  <div className="space-y-5">
                    {(usageStats?.team_usage ?? []).slice(0, 6).map((user) => (
                      <div key={user.user_id}>
                        <div className="flex items-center justify-between mb-2">
                          <span
                            className="text-white text-sm truncate max-w-[120px]"
                            title={
                              user.user_email || user.user_name || user.user_id
                            }
                          >
                            {user.user_name || user.user_email || "Unknown"}
                          </span>
                          <span className="text-[#888888] text-sm">
                            {user.percentage}%
                          </span>
                        </div>
                        <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-white rounded-full transition-all"
                            style={{ width: `${user.percentage}%` }}
                          />
                        </div>
                      </div>
                    ))}
                    {(!usageStats?.team_usage ||
                      usageStats.team_usage.length === 0) && (
                      <div className="text-[#888888] text-sm text-center py-4">
                        No usage data available
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </>
        ) : (
          <>
            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6 px-8 pt-6">
              <KPICard
                label="Active Conversations"
                value={stats?.active_conversations ?? "-"}
                subtext="In-flight agent work"
                icon={<ActivityIcon />}
              />
              <KPICard
                label="Running Runtimes"
                value={stats?.running_runtimes ?? "-"}
                subtext="Connected execution cells"
                icon={<ServerIcon />}
              />
              <KPICard
                label="Completed (24H)"
                value={stats?.completed_24h ?? "-"}
                subtext="Finished in last day"
                icon={<CheckCircleIcon />}
              />
              <KPICard
                label="Estimated Spend"
                value={formatCost(stats?.total_cost ?? 0)}
                subtext={`~${formatTokens(stats?.total_tokens ?? 0)} tokens in dataset`}
                icon={<DollarIcon />}
              />
            </div>

            {/* Filter Bar */}

            <div className="bg-[#161616] border border-[#262626] rounded-lg p-4 mb-4">
              <div className="flex items-center justify-between mb-4">
                <span className="text-[#8C8C8C] text-xs font-medium uppercase tracking-wide">
                  Filters
                </span>
                <a
                  href={exportUrl}
                  className="flex items-center gap-2 px-3 py-1.5 bg-[#1E1E1E] border border-[#262626] rounded text-sm text-[#8C8C8C] hover:text-white hover:border-[#404040] transition-colors"
                >
                  <ExportIcon />
                  Export CSV
                </a>
              </div>

              <div className="flex flex-wrap gap-3">
                {/* Search */}
                <div className="relative flex-1 min-w-[200px]">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-[#8C8C8C]">
                    <SearchIcon />
                  </div>
                  <input
                    type="text"
                    placeholder="Name, creator, email..."
                    value={search}
                    onChange={(e) => updateFilter("search", e.target.value)}
                    className="w-full pl-10 pr-4 py-2 bg-[#0D0D0D] border border-[#262626] rounded text-white text-sm placeholder-[#6B6B6B] focus:outline-none focus:border-[#404040]"
                  />
                </div>

                {/* Sort By */}
                <select
                  value={sortBy}
                  onChange={(e) => updateFilter("sort_by", e.target.value)}
                  className="px-3 py-2 bg-[#0D0D0D] border border-[#262626] rounded text-sm text-white focus:outline-none focus:border-[#404040] appearance-none cursor-pointer"
                >
                  <option value="updated_at">Last updated</option>
                  <option value="created_at">Created</option>
                  <option value="title">Title</option>
                  <option value="llm_model">Model</option>
                  <option value="accumulated_cost">Cost</option>
                </select>

                {/* Order */}
                <select
                  value={sortOrder}
                  onChange={(e) => updateFilter("sort_order", e.target.value)}
                  className="px-3 py-2 bg-[#0D0D0D] border border-[#262626] rounded text-sm text-white focus:outline-none focus:border-[#404040] appearance-none cursor-pointer"
                >
                  <option value="desc">Descending</option>
                  <option value="asc">Ascending</option>
                </select>

                {/* Conversation Status */}
                <select
                  value={executionStatus}
                  onChange={(e) =>
                    updateFilter("execution_status", e.target.value || null)
                  }
                  className="px-3 py-2 bg-[#0D0D0D] border border-[#262626] rounded text-sm text-white focus:outline-none focus:border-[#404040] appearance-none cursor-pointer"
                >
                  <option value="">Conversation status: All</option>
                  <option value="running">Running</option>
                  <option value="idle">Idle</option>
                  <option value="paused">Paused</option>
                  <option value="finished">Finished</option>
                  <option value="error">Error</option>
                  <option value="stuck">Stuck</option>
                </select>

                {/* Runtime Status */}
                <select
                  value={sandboxStatus}
                  onChange={(e) =>
                    updateFilter("sandbox_status", e.target.value || null)
                  }
                  className="px-3 py-2 bg-[#0D0D0D] border border-[#262626] rounded text-sm text-white focus:outline-none focus:border-[#404040] appearance-none cursor-pointer"
                >
                  <option value="">Runtime status: All</option>
                  <option value="RUNNING">Running</option>
                  <option value="STARTING">Starting</option>
                  <option value="PAUSED">Paused</option>
                  <option value="ERROR">Error</option>
                  <option value="MISSING">Missing</option>
                </select>

                {/* Time Window */}
                <select
                  value={timeWindow}
                  onChange={(e) =>
                    updateFilter("time_window", e.target.value || null)
                  }
                  className="px-3 py-2 bg-[#0D0D0D] border border-[#262626] rounded text-sm text-white focus:outline-none focus:border-[#404040] appearance-none cursor-pointer"
                >
                  <option value="">All time</option>
                  <option value="7d">Last 7 days</option>
                  <option value="30d">Last 30 days</option>
                  <option value="90d">Last 90 days</option>
                </select>
              </div>
            </div>

            {/* Table */}
            <div className="bg-[#161616] border border-[#262626] rounded-lg overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-[#262626]">
                      <th className="px-4 py-3 text-left text-[#8C8C8C] text-xs font-medium uppercase tracking-wide">
                        Conversation
                      </th>
                      <th className="px-4 py-3 text-left text-[#8C8C8C] text-xs font-medium uppercase tracking-wide">
                        Model
                      </th>
                      <th className="px-4 py-3 text-left text-[#8C8C8C] text-xs font-medium uppercase tracking-wide">
                        Created by
                      </th>
                      <th className="px-4 py-3 text-left text-[#8C8C8C] text-xs font-medium uppercase tracking-wide">
                        Status
                      </th>
                      <th className="px-4 py-3 text-left text-[#8C8C8C] text-xs font-medium uppercase tracking-wide">
                        Runtime
                      </th>
                      <th className="px-4 py-3 text-left text-[#8C8C8C] text-xs font-medium uppercase tracking-wide">
                        Runtime URL / GUID
                      </th>
                      <th className="px-4 py-3 text-left text-[#8C8C8C] text-xs font-medium uppercase tracking-wide">
                        Created / Updated
                      </th>
                      <th className="px-4 py-3 text-left text-[#8C8C8C] text-xs font-medium uppercase tracking-wide">
                        Tokens / Cost
                      </th>
                      <th className="px-4 py-3 text-left text-[#8C8C8C] text-xs font-medium uppercase tracking-wide">
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {conversationsLoading && (
                      <tr>
                        <td
                          colSpan={9}
                          className="px-4 py-8 text-center text-[#8C8C8C]"
                        >
                          Loading...
                        </td>
                      </tr>
                    )}
                    {!conversationsLoading &&
                      conversationsData?.items.length === 0 && (
                        <tr>
                          <td
                            colSpan={9}
                            className="px-4 py-8 text-center text-[#8C8C8C]"
                          >
                            No conversations found
                          </td>
                        </tr>
                      )}
                    {!conversationsLoading &&
                      conversationsData?.items.map((conversation, index) => (
                        <tr
                          key={conversation.id}
                          className={`border-b border-[#262626] hover:bg-[#1E1E1E]/50 transition-colors ${
                            index % 2 === 0 ? "" : "bg-[#0D0D0D]/30"
                          }`}
                        >
                          <td className="px-4 py-3">
                            <div className="font-medium text-white text-sm">
                              {conversation.title || "Untitled"}
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <span className="font-mono text-xs text-[#8C8C8C]">
                              {conversation.llm_model || "N/A"}
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            <div className="text-sm">
                              <div className="text-white">
                                {conversation.user_email?.split("@")[0] ||
                                  "Unknown"}
                              </div>
                              <div className="text-[#8C8C8C] text-xs">
                                {conversation.user_email || ""}
                              </div>
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <StatusBadge
                              status={conversation.execution_status}
                            />
                          </td>
                          <td className="px-4 py-3">
                            <RuntimeBadge
                              status={conversation.sandbox_status}
                            />
                          </td>
                          <td className="px-4 py-3">
                            {conversation.sandbox_id ? (
                              <div className="flex items-center gap-2">
                                <span className="font-mono text-xs text-[#8C8C8C] truncate max-w-[120px]">
                                  {conversation.sandbox_id.slice(0, 12)}...
                                </span>
                                <button
                                  type="button"
                                  onClick={() =>
                                    copyToClipboard(conversation.sandbox_id!)
                                  }
                                  className="text-[#6B6B6B] hover:text-white transition-colors"
                                  title="Copy GUID"
                                >
                                  <CopyIcon />
                                </button>
                                {conversation.runtime_url && (
                                  <a
                                    href={conversation.runtime_url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-[#6B6B6B] hover:text-white transition-colors"
                                    title="Open in new tab"
                                  >
                                    <ExternalLinkIcon />
                                  </a>
                                )}
                              </div>
                            ) : (
                              <span className="text-[#6B6B6B] text-xs">
                                N/A
                              </span>
                            )}
                          </td>
                          <td className="px-4 py-3">
                            <div className="text-xs">
                              <div className="text-[#8C8C8C] uppercase mb-0.5">
                                Created
                              </div>
                              <div className="text-white mb-1">
                                {formatDate(conversation.created_at)}
                              </div>
                              <div className="text-[#8C8C8C] uppercase mb-0.5">
                                Updated
                              </div>
                              <div className="text-white">
                                {formatDate(conversation.updated_at)}
                              </div>
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <div className="text-xs text-right">
                              <div className="text-white font-mono">
                                {formatTokens(conversation.total_tokens)}
                              </div>
                              <div className="text-white font-medium">
                                ${conversation.accumulated_cost.toFixed(2)}
                              </div>
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <div className="flex flex-col gap-1">
                              <button
                                type="button"
                                onClick={() => handleStop(conversation)}
                                disabled={
                                  !isStoppable(conversation.execution_status) ||
                                  stoppingIds.has(conversation.id)
                                }
                                className="flex items-center gap-1.5 px-2 py-1 text-xs text-[#8C8C8C] hover:text-white hover:bg-[#262626] rounded transition-colors disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-[#8C8C8C] disabled:cursor-not-allowed"
                                title={
                                  isStoppable(conversation.execution_status)
                                    ? "Stop conversation"
                                    : "Conversation is not running"
                                }
                                aria-label="Stop conversation"
                              >
                                <StopIcon />
                                {stoppingIds.has(conversation.id)
                                  ? "Stopping…"
                                  : "Stop"}
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination Footer */}
              <div className="flex items-center justify-between px-4 py-3 border-t border-[#262626]">
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => handlePageChange(page - 1)}
                    disabled={page <= 1}
                    className={`flex items-center gap-1 px-2 py-1 text-sm rounded transition-colors ${
                      page <= 1
                        ? "text-[#6B6B6B] cursor-not-allowed"
                        : "text-[#8C8C8C] hover:text-white hover:bg-[#262626]"
                    }`}
                  >
                    <ChevronLeftIcon />
                    Previous
                  </button>
                  <button
                    type="button"
                    onClick={() => handlePageChange(page + 1)}
                    disabled={page >= totalPages}
                    className={`flex items-center gap-1 px-2 py-1 text-sm rounded transition-colors ${
                      page >= totalPages
                        ? "text-[#6B6B6B] cursor-not-allowed"
                        : "text-[#8C8C8C] hover:text-white hover:bg-[#262626]"
                    }`}
                  >
                    Next
                    <ChevronRightIcon />
                  </button>
                </div>

                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <span className="text-[#8C8C8C] text-sm">Per page</span>
                    <select
                      value={perPage}
                      onChange={(e) => {
                        const newParams = new URLSearchParams(searchParams);
                        newParams.set("per_page", e.target.value);
                        newParams.set("page", "1");
                        setSearchParams(newParams);
                      }}
                      className="px-2 py-1 bg-[#0D0D0D] border border-[#262626] rounded text-sm text-white focus:outline-none"
                    >
                      <option value="5">5</option>
                      <option value="10">10</option>
                      <option value="20">20</option>
                      <option value="50">50</option>
                    </select>
                  </div>
                  <span className="text-[#8C8C8C] text-sm">
                    Page {page} of {totalPages} · {totalItems} conversations
                  </span>
                </div>
              </div>
            </div>
          </>
        )}
      </main>

      {pendingStop && (
        <ConfirmationModal
          text={stopConfirmationText}
          onConfirm={confirmStop}
          onCancel={cancelStop}
        />
      )}
    </div>
  );
}
