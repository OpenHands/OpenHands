/* eslint-disable i18next/no-literal-string */
import React, { useMemo, useState } from "react";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { useStopConversation } from "#/hooks/mutation/use-stop-conversation";
import { useOrgConversationStats } from "#/hooks/query/use-org-conversation-stats";
import { useOrgConversations } from "#/hooks/query/use-org-conversations";
import { useOrgUsageStats } from "#/hooks/query/use-org-usage-stats";
import { useOrgUserUsage } from "#/hooks/query/use-org-user-usage";
import { useOrganizations } from "#/hooks/query/use-organizations";
import { organizationService } from "#/api/organization-service/organization-service.api";
import {
  ConversationsTab,
  ModelsTab,
  OverviewTab,
  UsersTab,
} from "./usage-dashboard-tabs";
import {
  AGENT_COLORS,
  TIME_WINDOWS,
  formatCost,
} from "./usage-dashboard-utils";

// Tabs
const TABS = ["overview", "users", "models", "conversations"] as const;
type TabType = (typeof TABS)[number];

export function UsageDashboard() {
  const [activeTab, setActiveTab] = useState<TabType>("overview");
  const [timeWindow, setTimeWindow] = useState("30d");
  const [modelSearch, setModelSearch] = useState("");
  const [conversationSearch, setConversationSearch] = useState("");
  const [conversationStatus, setConversationStatus] = useState("running");

  const [conversationSortBy, setConversationSortBy] = useState("updated_at");
  const [conversationSortOrder, setConversationSortOrder] = useState("desc");
  const [conversationSandboxStatus, setConversationSandboxStatus] =
    useState("");

  const [conversationPage, setConversationPage] = useState(1);
  const [conversationPerPage, setConversationPerPage] = useState(20);

  const { organizationId } = useSelectedOrganizationId();
  const { data: orgData } = useOrganizations();

  const { data: stats } = useOrgConversationStats();
  const { data: usageStats } = useOrgUsageStats({ timeWindow });
  const { data: userUsage, isLoading: userUsageLoading } = useOrgUserUsage();

  const conversationTimeWindow = timeWindow === "ytd" ? "" : timeWindow;

  const { data: conversationsData, isLoading: conversationsLoading } =
    useOrgConversations({
      page: conversationPage,
      perPage: conversationPerPage,
      search: conversationSearch,
      sortBy: conversationSortBy,
      sortOrder: conversationSortOrder,
      executionStatus: conversationStatus,
      sandboxStatus: conversationSandboxStatus,
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

  const modelRows = useMemo(
    () =>
      (usageStats?.model_usage ?? []).map((model) => {
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
      }),
    [usageStats?.model_usage],
  );

  const filteredModels = useMemo(
    () =>
      modelRows.filter((model) =>
        model.model_name.toLowerCase().includes(modelSearch.toLowerCase()),
      ),
    [modelRows, modelSearch],
  );

  const chartData = useMemo(
    () =>
      (usageStats?.daily_usage ?? []).map((d) => ({
        date: d.date,
        value: d.conversations,
      })),
    [usageStats?.daily_usage],
  );

  const agentSpendRows = useMemo(() => {
    const rows = usageStats?.agent_usage ?? [];
    const total = rows.reduce((sum, row) => sum + row.total_cost, 0);
    return rows.map((row, index) => ({
      ...row,
      percent: total > 0 ? (row.total_cost / total) * 100 : 0,
      color: AGENT_COLORS[index % AGENT_COLORS.length],
    }));
  }, [usageStats?.agent_usage]);

  const agentSpendTotal = agentSpendRows.reduce(
    (sum, row) => sum + row.total_cost,
    0,
  );

  const tabCounts = {
    overview: null,
    users: userUsage?.items.length ?? 0,
    models: modelRows.length,
    conversations: conversationsData?.total_items ?? 0,
  };

  const timeWindowLabel =
    timeWindow === "ytd" ? "YTD" : timeWindow.toUpperCase();

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
      sortBy: conversationSortBy,
      sortOrder: conversationSortOrder,
      executionStatus: conversationStatus || undefined,
      sandboxStatus: conversationSandboxStatus || undefined,
      timeWindow: conversationTimeWindow || undefined,
    });
  }, [
    organizationId,
    conversationSearch,
    conversationSortBy,
    conversationSortOrder,
    conversationStatus,
    conversationSandboxStatus,
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
          <OverviewTab
            totalConversations={totalConversations}
            activeConversations={activeConversations}
            avgCostPerConversation={avgCostPerConversation}
            totalSpend={totalSpend}
            timeWindowLabel={timeWindowLabel}
            chartData={chartData}
            agentSpendRows={agentSpendRows}
            agentSpendTotal={agentSpendTotal}
          />
        )}

        {/* Conversations Tab */}
        {activeTab === "conversations" && (
          <ConversationsTab
            conversationSearch={conversationSearch}
            conversationStatus={conversationStatus}
            conversationSortBy={conversationSortBy}
            conversationSortOrder={conversationSortOrder}
            conversationSandboxStatus={conversationSandboxStatus}
            exportUrl={exportUrl}
            conversationPage={conversationPage}
            conversationPerPage={conversationPerPage}
            conversationTotalPages={conversationTotalPages}
            conversationTotalItems={conversationTotalItems}
            conversationsLoading={conversationsLoading}
            conversationsData={conversationsData}
            stoppingIds={stoppingIds}
            onSearchChange={(value) => {
              setConversationSearch(value);
              setConversationPage(1);
            }}
            onStatusChange={(value) => {
              setConversationStatus(value);
              setConversationPage(1);
            }}
            onSortByChange={(value) => {
              setConversationSortBy(value);
              setConversationPage(1);
            }}
            onSortOrderChange={(value) => {
              setConversationSortOrder(value);
              setConversationPage(1);
            }}
            onSandboxStatusChange={(value) => {
              setConversationSandboxStatus(value);
              setConversationPage(1);
            }}
            onPageChange={setConversationPage}
            onPerPageChange={(value) => {
              setConversationPerPage(value);
              setConversationPage(1);
            }}
            onStopConversation={handleStop}
            pendingStop={pendingStop}
            stopConfirmationText={stopConfirmationText}
            onConfirmStop={confirmStop}
            onCancelStop={cancelStop}
          />
        )}

        {/* Users Tab */}
        {activeTab === "users" && (
          <UsersTab userUsage={userUsage} userUsageLoading={userUsageLoading} />
        )}

        {/* Models Tab */}
        {activeTab === "models" && (
          <ModelsTab
            modelSearch={modelSearch}
            onModelSearchChange={setModelSearch}
            filteredModels={filteredModels}
          />
        )}
      </div>
    </div>
  );
}
