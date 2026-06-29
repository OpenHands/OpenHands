/* eslint-disable i18next/no-literal-string */
import React, { useMemo, useState } from "react";
import { useSearchParams } from "react-router";
import { useOrgConversationStats } from "#/hooks/query/use-org-conversation-stats";
import { useOrgConversations } from "#/hooks/query/use-org-conversations";
import { useOrgUsageStats } from "#/hooks/query/use-org-usage-stats";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { useOrganizations } from "#/hooks/query/use-organizations";

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

function ChevronDownIcon() {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <path d="M6 9l6 6 6-6" />
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

// Tabs
const TABS = ["overview", "conversations", "users", "models"] as const;
type TabType = (typeof TABS)[number];

// Sample model data (in real implementation, this would come from API)
const SAMPLE_MODELS = [
  {
    model: "claude-sonnet-4-5",
    provider: "Anthropic",
    conversations: 1124,
    tokensUsed: 84320000,
    avgTokensPerConvo: 75000,
    avgCostPerConvo: 1.64,
    totalCost: 1842.66,
  },
  {
    model: "claude-opus-4",
    provider: "Anthropic",
    conversations: 238,
    tokensUsed: 21410000,
    avgTokensPerConvo: 90000,
    avgCostPerConvo: 2.7,
    totalCost: 642.3,
  },
  {
    model: "gpt-5",
    provider: "OpenAI",
    conversations: 302,
    tokensUsed: 18120000,
    avgTokensPerConvo: 60000,
    avgCostPerConvo: 1.4,
    totalCost: 423.18,
  },
  {
    model: "gpt-5-mini",
    provider: "OpenAI",
    conversations: 108,
    tokensUsed: 4240000,
    avgTokensPerConvo: 39300,
    avgCostPerConvo: 0.35,
    totalCost: 38.16,
  },
  {
    model: "gemini-2.5-pro",
    provider: "Google",
    conversations: 54,
    tokensUsed: 3960000,
    avgTokensPerConvo: 73300,
    avgCostPerConvo: 1.47,
    totalCost: 79.2,
  },
  {
    model: "devstral-large",
    provider: "Mistral",
    conversations: 21,
    tokensUsed: 1180000,
    avgTokensPerConvo: 56200,
    avgCostPerConvo: 0.67,
    totalCost: 14.16,
  },
];

// Sample conversation data for detailed view
const SAMPLE_CONVERSATIONS = [
  {
    user: { name: "Tom Bauer", email: "tom.bauer@acme.com", initials: "TB", color: "bg-green-500" },
    tokens: "170.8K",
    spend: "$3.39",
    duration: "3h 31m",
    started: "Jun 27",
    lastUpdate: "Jun 27 11:36 PM",
    pr: { text: "acme/infra#2448", linked: true },
    merged: "Draft",
    mergedType: "draft",
    type: "Docs",
    typeColor: "bg-teal-500/20 text-teal-400 border-teal-500/30",
  },
  {
    user: { name: "Leila Ahmadi", email: "leila.ahmadi@acme.com", initials: "LA", color: "bg-purple-500" },
    tokens: "436.5K",
    spend: "$4.17",
    duration: "2h 39m",
    started: "Jun 25",
    lastUpdate: "Jun 25 6:52 PM",
    pr: { text: "No PR", linked: false },
    merged: "—",
    mergedType: "none",
    type: "Security",
    typeColor: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  },
  {
    user: { name: "Priya Shah", email: "priya.shah@acme.com", initials: "PS", color: "bg-amber-500" },
    tokens: "89.4K",
    spend: "$2.83",
    duration: "1h 30m",
    started: "Jun 25",
    lastUpdate: "Jun 25 4:21 PM",
    pr: { text: "acme/docs#9618", linked: true },
    merged: "Merged",
    mergedType: "merged",
    type: "Refactor",
    typeColor: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  },
  {
    user: { name: "Sam Osei", email: "sam.osei@acme.com", initials: "SO", color: "bg-purple-500" },
    tokens: "391.2K",
    spend: "$4.47",
    duration: "1h 30m",
    started: "Jun 25",
    lastUpdate: "Jun 25 7:48 AM",
    pr: { text: "acme/docs#2243", linked: true },
    merged: "Draft",
    mergedType: "draft",
    type: "Docs",
    typeColor: "bg-teal-500/20 text-teal-400 border-teal-500/30",
  },
  {
    user: { name: "DevOps Bot", email: "devops-bot@acme.com", initials: "DB", color: "bg-blue-500" },
    tokens: "256.6K",
    spend: "$2.62",
    duration: "2h 32m",
    started: "Jun 25",
    lastUpdate: "Jun 25 7:15 AM",
    pr: { text: "acme/web#8739", linked: true },
    merged: "Draft",
    mergedType: "draft",
    type: "Docs",
    typeColor: "bg-teal-500/20 text-teal-400 border-teal-500/30",
  },
  {
    user: { name: "Leila Ahmadi", email: "leila.ahmadi@acme.com", initials: "LA", color: "bg-purple-500" },
    tokens: "111.7K",
    spend: "$2.69",
    duration: "1h 3m",
    started: "Jun 19",
    lastUpdate: "Jun 19 10:20 AM",
    pr: { text: "No PR", linked: false },
    merged: "—",
    mergedType: "none",
    type: "Security",
    typeColor: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  },
  {
    user: { name: "Priya Shah", email: "priya.shah@acme.com", initials: "PS", color: "bg-amber-500" },
    tokens: "222.7K",
    spend: "$2.69",
    duration: "3h 7m",
    started: "Jun 17",
    lastUpdate: "Jun 17 3:38 PM",
    pr: { text: "No PR", linked: false },
    merged: "—",
    mergedType: "none",
    type: "Refactor",
    typeColor: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  },
  {
    user: { name: "Sam Osei", email: "sam.osei@acme.com", initials: "SO", color: "bg-purple-500" },
    tokens: "149.0K",
    spend: "$1.58",
    duration: "1h 10m",
    started: "Jun 17",
    lastUpdate: "Jun 17 6:04 AM",
    pr: { text: "acme/mobile#8358", linked: true },
    merged: "Open",
    mergedType: "open",
    type: "Refactor",
    typeColor: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  },
  {
    user: { name: "DevOps Bot", email: "devops-bot@acme.com", initials: "DB", color: "bg-blue-500" },
    tokens: "32.5K",
    spend: "$2.59",
    duration: "1h 57m",
    started: "Jun 17",
    lastUpdate: "Jun 17 3:21 AM",
    pr: { text: "acme/infra#4415", linked: true },
    merged: "Draft",
    mergedType: "draft",
    type: "Refactor",
    typeColor: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  },
];

// Merged status styles
function MergedBadge({ status }: { status: string; type: string }) {
  if (type === "none") {
    return <span className="text-zinc-600">—</span>;
  }
  if (type === "merged") {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-green-500/20 text-green-400 border border-green-500/30">
        ✓ {status}
      </span>
    );
  }
  if (type === "draft") {
    return (
      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-orange-500/20 text-orange-400 border border-orange-500/30">
        {status}
      </span>
    );
  }
  if (type === "open") {
    return (
      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-500/20 text-blue-400 border border-blue-500/30">
        {status}
      </span>
    );
  }
  return <span className="text-zinc-400">{status}</span>;
}

// Avatar component
function Avatar({ initials, color }: { initials: string; color: string }) {
  return (
    <div className={`w-8 h-8 rounded-full ${color} flex items-center justify-center text-xs font-medium text-white`}>
      {initials}
    </div>
  );
}

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
        <div className={`flex items-center gap-1 mt-2 text-xs ${trendUp ? "text-green-400" : "text-red-400"}`}>
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
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full" preserveAspectRatio="none">
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
        {data.filter((_, i) => i % Math.ceil(data.length / 7) === 0).map((d) => (
          <span key={d.date}>{formatShortDate(d.date)}</span>
        ))}
      </div>
    </div>
  );
}

export function UsageDashboard() {
  const [activeTab, setActiveTab] = useState<TabType>("overview");
  const [timeWindow, setTimeWindow] = useState("30d");
  const [searchParams, setSearchParams] = useSearchParams();
  const [modelSearch, setModelSearch] = useState("");

  const { organizationId } = useSelectedOrganizationId();
  const { data: orgData } = useOrganizations();

  const days = getDaysFromTimeWindow(timeWindow);
  const { data: stats } = useOrgConversationStats();
  const { data: usageStats } = useOrgUsageStats({ days });

  const currentOrg = orgData?.organizations?.find((org) => org.id === organizationId);

  // Calculate totals from stats for overview
  const totalConversations = stats?.completed_30d ?? 1847;
  const activeConversations = stats?.active_conversations ?? 23;
  const avgCostPerConversation = stats && stats.completed_30d > 0
    ? (stats.total_cost / stats.completed_30d).toFixed(2)
    : "1.42";
  const totalSpend = formatCost(stats?.total_cost ?? 2623.47);

  // Filter models based on search
  const filteredModels = SAMPLE_MODELS.filter(
    (m) =>
      m.model.toLowerCase().includes(modelSearch.toLowerCase()) ||
      m.provider.toLowerCase().includes(modelSearch.toLowerCase())
  );

  // Generate chart data from usage stats
  const chartData = useMemo(() => {
    return (usageStats?.daily_usage ?? []).map((d) => ({
      date: d.date,
      value: d.tokens,
    }));
  }, [usageStats?.daily_usage]);

  // Tab badge counts
  const tabCounts = {
    overview: null,
    conversations: stats?.completed_30d ?? 1847,
    users: usageStats?.active_users ?? 8,
    models: SAMPLE_MODELS.length,
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-white">
      {/* Header */}
      <div className="px-8 py-6 border-b border-zinc-800">
        <div className="flex items-start justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-white mb-1">Usage Dashboard</h1>
            <p className="text-zinc-400">
              Monitor adoption, spend, and ROI across your organization.
            </p>
          </div>
          {/* Time window selector */}
          <div className="flex items-center gap-1 bg-zinc-900 border border-zinc-800 rounded-lg p-1">
            {TIME_WINDOWS.map((tw) => (
              <button
                key={tw.value}
                type="button"
                onClick={() => setTimeWindow(tw.value)}
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
              {tabCounts[tab] && (
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
                value="1,847"
                trend="▲ 18.2% vs prev 30d"
                trendUp
              />
              <KPICard
                label="Active Conversations"
                value="23"
                trend="▲ 4 more than yesterday"
                trendUp
              />
              <KPICard
                label="Avg Cost / Conversation"
                value="$1.42"
                trend="▼ 6.1% vs prev 30d"
                trendUp={false}
              />
              <KPICard
                label="Total Spend (30d)"
                value="$2,623.47"
                trend="▲ 11.4% vs prev 30d"
                trendUp
              />
            </div>

            {/* Chart Row */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6">
              <div className="flex items-start justify-between mb-6">
                <div>
                  <h2 className="text-lg font-medium text-white">Conversations per day</h2>
                  <p className="text-sm text-zinc-500">Last 30 days · all users</p>
                </div>
                <button
                  type="button"
                  className="flex items-center gap-2 px-3 py-1.5 text-sm text-zinc-400 border border-zinc-700 rounded-lg hover:text-white hover:border-zinc-600 transition-colors"
                >
                  <ExportIcon />
                  Export CSV
                </button>
              </div>
              <AreaChart data={chartData.length > 0 ? chartData : [
                { date: "2024-05-31", value: 45 },
                { date: "2024-06-04", value: 52 },
                { date: "2024-06-08", value: 38 },
                { date: "2024-06-12", value: 65 },
                { date: "2024-06-16", value: 48 },
                { date: "2024-06-20", value: 55 },
                { date: "2024-06-24", value: 42 },
                { date: "2024-06-28", value: 58 },
              ]} />
            </div>

            {/* ROI Row */}
            <div className="grid grid-cols-3 gap-4">
              <KPICard
                label="Engineering Velocity Gained"
                value="412h"
                trend="est. dev-hours saved (30d)"
                trendUp
              />
              <KPICard
                label="Cost / Dev-Hour Saved"
                value="$6.37"
                trend="▲ 9.0× more efficient vs human"
                trendUp
              />
              <KPICard
                label="PRs Merged with OpenHands"
                value="214"
                trend="▲ 62% merge rate"
                trendUp
              />
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
                  placeholder="Search by user, repo, or PR..."
                  className="w-full pl-10 pr-4 py-2 bg-zinc-900 border border-zinc-800 rounded-lg text-white placeholder-zinc-500 focus:outline-none focus:border-zinc-700"
                />
              </div>
              <div className="flex items-center gap-2">
                {["Type: All", "Status: All", "User: All"].map((filter) => (
                  <button
                    key={filter}
                    type="button"
                    className="flex items-center gap-1 px-3 py-2 text-sm text-zinc-400 bg-zinc-900 border border-zinc-800 rounded-lg hover:text-white hover:border-zinc-700 transition-colors"
                  >
                    {filter}
                    <ChevronDownIcon />
                  </button>
                ))}
              </div>
              <button
                type="button"
                className="flex items-center gap-2 px-3 py-2 text-sm text-zinc-400 border border-zinc-700 rounded-lg hover:text-white hover:border-zinc-600 transition-colors"
              >
                <ExportIcon />
                Export CSV
              </button>
            </div>

            {/* Conversations Table */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-zinc-800">
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">User</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">Tokens</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">Spend</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">Duration</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">Started</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">Last Update</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">Associated PR</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">Merged?</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">Type</th>
                  </tr>
                </thead>
                <tbody>
                  {SAMPLE_CONVERSATIONS.map((conv, i) => (
                    <tr key={i} className="border-b border-zinc-800/50 hover:bg-zinc-800/50 transition-colors">
                      <td className="px-4 py-4">
                        <div className="flex items-center gap-3">
                          <Avatar initials={conv.user.initials} color={conv.user.color} />
                          <span className="text-zinc-400 text-sm">{conv.user.email}</span>
                        </div>
                      </td>
                      <td className="px-4 py-4 text-white text-sm font-mono">{conv.tokens}</td>
                      <td className="px-4 py-4 text-white text-sm font-mono">{conv.spend}</td>
                      <td className="px-4 py-4 text-white text-sm">{conv.duration}</td>
                      <td className="px-4 py-4 text-zinc-400 text-sm">{conv.started}</td>
                      <td className="px-4 py-4 text-zinc-400 text-sm">{conv.lastUpdate}</td>
                      <td className="px-4 py-4">
                        {conv.pr.linked ? (
                          <a href="#" className="text-blue-400 hover:text-blue-300 text-sm">
                            {conv.pr.text}
                          </a>
                        ) : (
                          <span className="text-zinc-600 text-sm">{conv.pr.text}</span>
                        )}
                      </td>
                      <td className="px-4 py-4">
                        <MergedBadge status={conv.merged} type={conv.mergedType} />
                      </td>
                      <td className="px-4 py-4">
                        <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${conv.typeColor}`}>
                          {conv.type}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Users Tab */}
        {activeTab === "users" && (
          <div className="space-y-4">
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6">
              <h2 className="text-lg font-medium text-white mb-6">Team Usage Breakdown</h2>
              <div className="space-y-4">
                {(usageStats?.team_usage ?? [
                  { user_name: "Tom Bauer", user_email: "tom.bauer@acme.com", conversation_count: 342, total_tokens: 45230000, percentage: 38.2 },
                  { user_name: "Leila Ahmadi", user_email: "leila.ahmadi@acme.com", conversation_count: 256, total_tokens: 32100000, percentage: 27.1 },
                  { user_name: "Priya Shah", user_email: "priya.shah@acme.com", conversation_count: 189, total_tokens: 18750000, percentage: 15.8 },
                  { user_name: "Sam Osei", user_email: "sam.osei@acme.com", conversation_count: 134, total_tokens: 14320000, percentage: 12.1 },
                  { user_name: "DevOps Bot", user_email: "devops-bot@acme.com", conversation_count: 89, total_tokens: 8230000, percentage: 6.8 },
                ]).map((user, i) => (
                  <div key={i} className="flex items-center gap-4">
                    <div className="w-32">
                      <div className="text-white text-sm font-medium truncate">
                        {user.user_name ?? user.user_email?.split("@")[0] ?? "Unknown"}
                      </div>
                      <div className="text-zinc-500 text-xs truncate">{user.user_email}</div>
                    </div>
                    <div className="flex-1">
                      <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-blue-500 rounded-full"
                          style={{ width: `${user.percentage}%` }}
                        />
                      </div>
                    </div>
                    <div className="w-24 text-right">
                      <div className="text-white text-sm">{formatTokens(user.total_tokens)}</div>
                      <div className="text-zinc-500 text-xs">{user.conversation_count} convos</div>
                    </div>
                    <div className="w-16 text-right">
                      <span className="text-zinc-400 text-sm">{user.percentage}%</span>
                    </div>
                  </div>
                ))}
              </div>
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
                    <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">Model</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase tracking-wider">Conversations</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase tracking-wider">Tokens Used</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase tracking-wider">Avg Tokens / Convo</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase tracking-wider">Avg Cost / Convo</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 uppercase tracking-wider">Total Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredModels.map((model, i) => (
                    <tr key={i} className="border-b border-zinc-800/50 hover:bg-zinc-800/50 transition-colors">
                      <td className="px-4 py-5">
                        <div className="text-white font-medium">{model.model}</div>
                        <div className="text-zinc-500 text-sm">{model.provider}</div>
                      </td>
                      <td className="px-4 py-5 text-white text-sm font-mono text-right">{model.conversations.toLocaleString()}</td>
                      <td className="px-4 py-5 text-white text-sm font-mono text-right">{formatTokens(model.tokensUsed)}</td>
                      <td className="px-4 py-5 text-white text-sm font-mono text-right">{formatTokens(model.avgTokensPerConvo)}</td>
                      <td className="px-4 py-5 text-white text-sm font-mono text-right">${model.avgCostPerConvo.toFixed(2)}</td>
                      <td className="px-4 py-5 text-white text-sm font-mono text-right font-medium">${model.totalCost.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
