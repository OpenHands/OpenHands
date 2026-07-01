/* eslint-disable i18next/no-literal-string */
import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { useConfig } from "#/hooks/query/use-config";

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

function EmailIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
      <polyline points="22,6 12,13 2,6" />
    </svg>
  );
}

function SlackIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <path d="M14.5 10c-.83 0-1.5-.67-1.5-1.5v-5c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5v5c0 .83-.67 1.5-1.5 1.5z" />
      <path d="M20.5 10H19V8.5c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5-.67 1.5-1.5 1.5z" />
      <path d="M9.5 14c.83 0 1.5.67 1.5 1.5v5c0 .83-.67 1.5-1.5 1.5S8 21.33 8 20.5v-5c0-.83.67-1.5 1.5-1.5z" />
      <path d="M3.5 14H5v1.5c0 .83-.67 1.5-1.5 1.5S2 16.33 2 15.5 2.67 14 3.5 14z" />
      <path d="M14 14.5c0-.83.67-1.5 1.5-1.5h5c.83 0 1.5.67 1.5 1.5s-.67 1.5-1.5 1.5h-5c-.83 0-1.5-.67-1.5-1.5z" />
      <path d="M15.5 19H14v1.5c0 .83.67 1.5 1.5 1.5s1.5-.67 1.5-1.5-.67-1.5-1.5-1.5z" />
      <path d="M10 9.5C10 8.67 9.33 8 8.5 8h-5C2.67 8 2 8.67 2 9.5S2.67 11 3.5 11h5c.83 0 1.5-.67 1.5-1.5z" />
      <path d="M8.5 5H10V3.5C10 2.67 9.33 2 8.5 2S7 2.67 7 3.5 7.67 5 8.5 5z" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <polyline points="3,6 5,6 21,6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </svg>
  );
}

function HashIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <line x1="4" y1="9" x2="20" y2="9" />
      <line x1="4" y1="15" x2="20" y2="15" />
      <line x1="10" y1="3" x2="8" y2="21" />
      <line x1="16" y1="3" x2="14" y2="21" />
    </svg>
  );
}

// Toggle component
function Toggle({
  enabled,
  onChange,
  label,
}: {
  enabled: boolean;
  onChange: (value: boolean) => void;
  label: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={enabled}
      aria-label={label}
      onClick={() => onChange(!enabled)}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
        enabled ? "bg-blue-500" : "bg-[#262626]"
      }`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
          enabled ? "translate-x-6" : "translate-x-1"
        }`}
      />
    </button>
  );
}

// Pill badge component
function PillBadge({
  active,
  icon,
  label,
  disabled = false,
}: {
  active: boolean;
  icon: React.ReactNode;
  label: string;
  disabled?: boolean;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border transition-colors ${
        active
          ? "bg-blue-500/10 text-blue-400 border-blue-500/30"
          : "bg-[#151D2A] text-[#6B6B6B] border-[#262626]"
      } ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
    >
      {active && <span className="text-blue-400">✓</span>}
      {icon}
      {label}
    </span>
  );
}

// Progress bar with gradient
function SpendMeter({
  percentage,
  showTicks = true,
}: {
  percentage: number;
  showTicks?: boolean;
}) {
  const getBarColor = () => {
    if (percentage >= 90)
      return "bg-gradient-to-r from-green-500 via-yellow-500 to-red-500";
    if (percentage >= 80)
      return "bg-gradient-to-r from-green-500 via-yellow-500 to-orange-500";
    return "bg-gradient-to-r from-green-500 to-yellow-500";
  };

  return (
    <div className="w-full">
      <div className="relative w-full h-3 bg-[#0B0F17] rounded-full overflow-hidden">
        <div
          className={`absolute inset-y-0 left-0 rounded-full ${getBarColor()}`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
      {showTicks && (
        <div className="relative mt-1">
          <div className="flex justify-between text-[10px] text-[#6B6B6B]">
            <span>0%</span>
            <span>80%</span>
            <span>90%</span>
            <span>100%</span>
          </div>
          {/* Tick marks */}
          <div className="absolute top-0 left-[80%] w-px h-2 bg-[#6B6B6B]" />
          <div className="absolute top-0 left-[90%] w-px h-2 bg-[#6B6B6B]" />
          <div className="absolute top-0 left-[100%] w-px h-2 bg-[#6B6B6B]" />
        </div>
      )}
    </div>
  );
}

// User progress bar
function UserProgressBar({
  value,
  max,
  status,
}: {
  value: number;
  max: number;
  status: "green" | "yellow" | "red";
}) {
  const percentage = max > 0 ? (value / max) * 100 : 0;
  const colorClass = {
    red: "bg-red-500",
    yellow: "bg-yellow-500",
    green: "bg-green-500",
  }[status];

  return (
    <div className="w-full">
      <div className="relative w-full h-1.5 bg-[#0B0F17] rounded-full overflow-hidden">
        <div
          className={`absolute inset-y-0 left-0 rounded-full ${colorClass}`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
      <div className="text-xs text-[#8C8C8C] mt-1">
        ${value.toLocaleString()} / ${max.toLocaleString()}
      </div>
    </div>
  );
}

// Avatar component
function Avatar({ name, size = "md" }: { name: string; size?: "sm" | "md" }) {
  const initials = name
    .split(" ")
    .map((n) => n[0])
    .join("")
    .toUpperCase()
    .slice(0, 2);

  const sizeClass = size === "sm" ? "w-7 h-7 text-xs" : "w-9 h-9 text-sm";

  return (
    <div
      className={`${sizeClass} rounded-full bg-[#262626] text-white flex items-center justify-center font-medium`}
    >
      {initials}
    </div>
  );
}

// Status pill for user table
function StatusPill({ status }: { status: string }) {
  const getStyle = () => {
    if (status.includes("Over cap")) {
      return "bg-red-500/20 text-red-400 border-red-500/30";
    }
    if (status.includes("> 90%")) {
      return "bg-red-500/10 text-red-400 border-red-500/30";
    }
    if (status.includes("> 80%")) {
      return "bg-yellow-500/10 text-yellow-400 border-yellow-500/30";
    }
    if (status.includes("On track")) {
      return "bg-green-500/10 text-green-400 border-green-500/30";
    }
    return "bg-[#151D2A] text-[#6B6B6B] border-[#262626]";
  };

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${getStyle()}`}
    >
      {status}
    </span>
  );
}

const BUDGET_TABS = [
  { value: "organization", label: "Organization budget" },
  { value: "defaults", label: "Default budgets" },
  { value: "overrides", label: "User overrides" },
] as const;

type BudgetTab = (typeof BUDGET_TABS)[number]["value"];

export function Budgets() {
  const { organizationId } = useSelectedOrganizationId();
  const queryClient = useQueryClient();

  const { data: config } = useConfig();
  const slackIntegrationEnabled = Boolean(config?.slack_enabled);
  const emailIntegrationEnabled = Boolean(config?.email_enabled);

  const [activeTab, setActiveTab] = useState<BudgetTab>("organization");

  const { data: budgetData, isLoading } = useQuery({
    queryKey: ["organizations", "budgets", organizationId],
    queryFn: () =>
      organizationService.getBudgetSettings({
        orgId: organizationId!,
      }),
    enabled: !!organizationId,
  });

  const updateBudgets = useMutation({
    mutationFn: (payload: {
      enabled?: boolean | null;
      monthly_limit?: number | null;
      reset_day?: number | null;
      default_user_monthly_limit?: number | null;
      slack_channel?: string | null;
      slack_team_id?: string | null;
      thresholds?:
        | {
            percentage: number;
            email_enabled: boolean;
            slack_enabled: boolean;
          }[]
        | null;
    }) =>
      organizationService.updateBudgetSettings({
        orgId: organizationId!,
        payload,
      }),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: ["organizations", "budgets", organizationId],
      }),
  });

  const upsertOverride = useMutation({
    mutationFn: (params: {
      userId: string;
      payload: { monthly_limit?: number | null; is_disabled: boolean };
    }) =>
      organizationService.upsertBudgetOverride({
        orgId: organizationId!,
        userId: params.userId,
        payload: params.payload,
      }),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: ["organizations", "budgets", organizationId],
      }),
  });

  const deleteOverride = useMutation({
    mutationFn: (userId: string) =>
      organizationService.deleteBudgetOverride({
        orgId: organizationId!,
        userId,
      }),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: ["organizations", "budgets", organizationId],
      }),
  });

  const [orgBudgetEnabled, setOrgBudgetEnabled] = useState(false);
  const [monthlyLimit, setMonthlyLimit] = useState("");
  const [billingCycle, setBillingCycle] = useState("1st");
  const [slackChannel, setSlackChannel] = useState("");
  const [thresholds, setThresholds] = useState<
    { percentage: number; email_enabled: boolean; slack_enabled: boolean }[]
  >([]);
  const [defaultAmount, setDefaultAmount] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [editingUserId, setEditingUserId] = useState<string | null>(null);
  const [overrideAmount, setOverrideAmount] = useState("");
  const [overrideDisabled, setOverrideDisabled] = useState(false);

  useEffect(() => {
    if (!budgetData) return;
    setOrgBudgetEnabled(budgetData.enabled);
    setMonthlyLimit(
      budgetData.monthly_limit ? budgetData.monthly_limit.toString() : "",
    );
    setBillingCycle(budgetData.reset_day === 15 ? "15th" : "1st");
    setSlackChannel(budgetData.slack_channel ?? "");
    setThresholds(
      budgetData.thresholds.map((threshold) => ({
        percentage: threshold.percentage,
        email_enabled: threshold.email_enabled,
        slack_enabled: threshold.slack_enabled,
      })),
    );
    setDefaultAmount(
      budgetData.default_user_monthly_limit
        ? budgetData.default_user_monthly_limit.toString()
        : "",
    );
  }, [budgetData]);

  const monthlyLimitValue = monthlyLimit ? Number(monthlyLimit) : null;
  const isMonthlyLimitValid =
    !orgBudgetEnabled ||
    (typeof monthlyLimitValue === "number" && monthlyLimitValue > 0);

  const currentSpend = budgetData?.current_spend ?? 0;
  const percentage = budgetData?.current_spend_percentage ?? 0;
  const cycleLabel = budgetData?.cycle_start_at
    ? new Date(budgetData.cycle_start_at).toLocaleDateString("en-US", {
        month: "long",
      })
    : "this cycle";
  const defaultUserLimit = budgetData?.default_user_monthly_limit ?? null;

  const defaultAmountLabel = defaultAmount
    ? parseFloat(defaultAmount).toLocaleString()
    : "0";

  const handleReset = () => {
    if (!budgetData) return;
    setOrgBudgetEnabled(budgetData.enabled);
    setMonthlyLimit(
      budgetData.monthly_limit ? budgetData.monthly_limit.toString() : "",
    );
    setBillingCycle(budgetData.reset_day === 15 ? "15th" : "1st");
    setSlackChannel(budgetData.slack_channel ?? "");
    setThresholds(
      budgetData.thresholds.map((threshold) => ({
        percentage: threshold.percentage,
        email_enabled: threshold.email_enabled,
        slack_enabled: threshold.slack_enabled,
      })),
    );
    setDefaultAmount(
      budgetData.default_user_monthly_limit
        ? budgetData.default_user_monthly_limit.toString()
        : "",
    );
  };

  const handleSaveOrgBudget = () => {
    if (!organizationId || !isMonthlyLimitValid) return;
    updateBudgets.mutate({
      enabled: orgBudgetEnabled,
      monthly_limit: monthlyLimitValue,
      reset_day: billingCycle === "15th" ? 15 : 1,
      slack_channel: slackIntegrationEnabled
        ? slackChannel.trim() || null
        : null,
      thresholds: thresholds.map((threshold) => ({
        percentage: threshold.percentage,
        email_enabled: emailIntegrationEnabled
          ? threshold.email_enabled
          : false,
        slack_enabled: slackIntegrationEnabled
          ? threshold.slack_enabled
          : false,
      })),
    });
  };

  const handleSaveDefault = () => {
    if (!organizationId) return;
    const defaultValue = defaultAmount ? Number(defaultAmount) : null;
    updateBudgets.mutate({
      default_user_monthly_limit: defaultValue,
    });
  };

  const handleAddThreshold = () => {
    const used = new Set(thresholds.map((t) => t.percentage));
    const candidates = [50, 60, 70, 75, 85, 95];
    const next = candidates.find((value) => !used.has(value));
    if (!next) return;
    setThresholds((prev) =>
      [
        ...prev,
        { percentage: next, email_enabled: true, slack_enabled: false },
      ].sort((a, b) => a.percentage - b.percentage),
    );
  };

  const handleDeleteThreshold = (index: number) => {
    setThresholds(thresholds.filter((_, i) => i !== index));
  };

  const handleToggleEmail = (index: number) => {
    if (!emailIntegrationEnabled) return;
    setThresholds(
      thresholds.map((t, i) =>
        i === index ? { ...t, email_enabled: !t.email_enabled } : t,
      ),
    );
  };

  const handleToggleSlack = (index: number) => {
    if (!slackIntegrationEnabled) return;
    setThresholds(
      thresholds.map((t, i) =>
        i === index ? { ...t, slack_enabled: !t.slack_enabled } : t,
      ),
    );
  };

  const userRows = useMemo(
    () =>
      (budgetData?.users ?? []).map((user) => {
        const limit = user.is_disabled ? null : user.effective_monthly_limit;
        const hasLimit = typeof limit === "number" && limit > 0;
        const usagePercent = hasLimit ? (user.current_spend / limit) * 100 : 0;
        let status = "No cap";
        let statusColor: "green" | "yellow" | "red" = "green";

        if (user.is_disabled) {
          status = "Disabled";
        } else if (hasLimit) {
          if (usagePercent > 100) {
            status = "Over cap";
            statusColor = "red";
          } else if (usagePercent >= 90) {
            status = "> 90% used";
            statusColor = "red";
          } else if (usagePercent >= 80) {
            status = "> 80% used";
            statusColor = "yellow";
          } else {
            status = "On track";
            statusColor = "green";
          }
        }

        let budgetLabel = "No limit";
        if (user.is_disabled) {
          budgetLabel = "Disabled";
        } else if (hasLimit) {
          budgetLabel = `$${limit.toLocaleString()} / month`;
        }

        let budgetNote = "No default";
        if (user.is_override) {
          budgetNote = "Override";
        } else if (defaultUserLimit) {
          budgetNote = "Inherits default";
        }

        return {
          ...user,
          name: user.user_name || user.user_email || "Unknown user",
          email: user.user_email || "",
          hasLimit,
          budgetLabel,
          budgetNote,
          status,
          statusColor,
          usage: user.current_spend,
          maxUsage: limit ?? 0,
        };
      }),
    [budgetData, defaultUserLimit],
  );

  const filteredUsers = useMemo(
    () =>
      userRows.filter((user) => {
        const matchesSearch =
          user.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          user.email.toLowerCase().includes(searchQuery.toLowerCase());
        const matchesStatus =
          statusFilter === "all" ||
          (statusFilter === "over80" &&
            (user.status.includes("> 80%") ||
              user.status.includes("> 90%") ||
              user.status.includes("Over cap"))) ||
          (statusFilter === "over90" &&
            (user.status.includes("> 90%") ||
              user.status.includes("Over cap"))) ||
          (statusFilter === "overCap" && user.status.includes("Over cap")) ||
          (statusFilter === "onTrack" && user.status.includes("On track")) ||
          (statusFilter === "noCap" && user.status.includes("No cap")) ||
          (statusFilter === "disabled" && user.status.includes("Disabled"));
        return matchesSearch && matchesStatus;
      }),
    [userRows, searchQuery, statusFilter],
  );

  const startEditingUser = (user: (typeof userRows)[number]) => {
    setEditingUserId(user.user_id);
    setOverrideDisabled(user.is_disabled);
    setOverrideAmount(
      user.effective_monthly_limit
        ? user.effective_monthly_limit.toString()
        : "",
    );
  };

  const cancelEditing = () => {
    setEditingUserId(null);
    setOverrideAmount("");
    setOverrideDisabled(false);
  };

  const saveOverride = (userId: string) => {
    if (!organizationId) return;
    const overrideValue = overrideAmount ? Number(overrideAmount) : null;
    upsertOverride.mutate({
      userId,
      payload: {
        monthly_limit: overrideDisabled ? null : overrideValue,
        is_disabled: overrideDisabled,
      },
    });
    cancelEditing();
  };

  const removeOverride = (userId: string) => {
    if (!organizationId) return;
    deleteOverride.mutate(userId);
  };

  if (!organizationId) {
    return (
      <div className="text-[#8C8C8C]">
        Select an organization to manage budgets.
      </div>
    );
  }

  if (isLoading) {
    return <div className="text-[#8C8C8C]">Loading budgets...</div>;
  }

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-semibold text-white mb-1">
            Budget settings
          </h1>
          <p className="text-[#8C8C8C]">
            Control your AI spend at the organization and user level.
          </p>
        </div>
        <div className="flex gap-6 border-b border-[#262626]">
          {BUDGET_TABS.map((tab) => (
            <button
              key={tab.value}
              type="button"
              onClick={() => setActiveTab(tab.value)}
              className={`flex items-center px-1 py-3 text-sm font-medium transition-colors border-b-2 ${
                activeTab === tab.value
                  ? "border-blue-500 text-white"
                  : "border-transparent text-[#8C8C8C] hover:text-white"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {activeTab === "organization" && (
        <div className="bg-[#151D2A] border border-[#262626] rounded-lg p-6">
          <div className="flex items-start justify-between mb-6">
            <div>
              <h2 className="text-lg font-medium text-white mb-1">
                Organization monthly budget
              </h2>
              <p className="text-sm text-[#8C8C8C]">
                Track total spend across your org and get alerted before you hit
                your cap.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-sm text-[#8C8C8C]">Enable budget</span>
              <Toggle
                enabled={orgBudgetEnabled}
                onChange={setOrgBudgetEnabled}
                label="Enable organization budget"
              />
            </div>
          </div>

          {/* Spend Meter */}
          <div className="mb-6">
            <div className="flex items-baseline justify-between mb-3">
              <div>
                <span className="text-3xl font-bold text-white">
                  $
                  {currentSpend.toLocaleString("en-US", {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </span>
                <span className="text-[#8C8C8C] ml-2">
                  {monthlyLimitValue
                    ? `of $${monthlyLimitValue.toLocaleString()} spent in ${cycleLabel}`
                    : `spent in ${cycleLabel}`}
                </span>
              </div>
              <span className="text-xl font-semibold text-yellow-400">
                {monthlyLimitValue ? `${percentage.toFixed(1)}%` : "—"}
              </span>
            </div>
            <SpendMeter percentage={percentage} />
          </div>

          {/* Inputs */}
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div>
              <label
                htmlFor="org-monthly-limit"
                className="block text-sm text-[#8C8C8C] mb-2"
              >
                Monthly limit
              </label>
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[#6B6B6B]">
                  $
                </span>
                <input
                  id="org-monthly-limit"
                  type="number"
                  value={monthlyLimit}
                  onChange={(e) => setMonthlyLimit(e.target.value)}
                  className="w-full pl-7 pr-4 py-2 bg-[#0B0F17] border border-[#262626] rounded-lg text-white focus:outline-none focus:border-blue-500"
                />
              </div>
            </div>
            <div>
              <label
                htmlFor="org-billing-cycle"
                className="block text-sm text-[#8C8C8C] mb-2"
              >
                Billing cycle resets
              </label>
              <select
                id="org-billing-cycle"
                value={billingCycle}
                onChange={(e) => setBillingCycle(e.target.value)}
                className="w-full px-4 py-2 bg-[#0B0F17] border border-[#262626] rounded-lg text-white focus:outline-none focus:border-blue-500"
              >
                <option value="1st">1st of each month</option>
                <option value="15th">15th of each month</option>
              </select>
            </div>
          </div>

          {/* Alert Thresholds */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="text-sm font-medium text-white mb-1">
                  Alert thresholds
                </h3>
                <p className="text-xs text-[#6B6B6B]">
                  Add one or more thresholds. Each can email admins, post to
                  Slack, or both.
                </p>
                {(!emailIntegrationEnabled || !slackIntegrationEnabled) && (
                  <div className="mt-2 space-y-1 text-xs text-amber-400">
                    {!emailIntegrationEnabled && (
                      <p>
                        Email alerts require RESEND_API_KEY set in the
                        deployment environment and a restart.
                      </p>
                    )}
                    {!slackIntegrationEnabled && (
                      <p>
                        Slack alerts require the Slack app to be configured in
                        the deployment (SLACK_* env vars). After a restart,
                        connect it in{" "}
                        <Link
                          to="/settings/integrations"
                          className="underline underline-offset-2"
                        >
                          Settings → Integrations
                        </Link>
                        .
                      </p>
                    )}
                  </div>
                )}
              </div>
              <button
                type="button"
                onClick={handleAddThreshold}
                className="px-3 py-1.5 text-sm text-blue-400 border border-blue-400/30 rounded-lg hover:bg-blue-500/10 transition-colors"
              >
                + Add threshold
              </button>
            </div>

            {/* Threshold rows */}
            <div className="space-y-3">
              {thresholds.map((threshold, index) => {
                const thresholdAmount = monthlyLimitValue
                  ? (monthlyLimitValue * threshold.percentage) / 100
                  : null;
                return (
                  <div
                    key={threshold.percentage}
                    className="flex items-center gap-4 p-3 bg-[#0B0F17] rounded-lg border border-[#262626]"
                  >
                    <div className="w-16">
                      <span className="text-white font-medium">
                        {threshold.percentage}%
                      </span>
                    </div>
                    <div className="w-28">
                      <span className="text-[#8C8C8C] text-sm">
                        {thresholdAmount !== null
                          ? `Triggers at $${thresholdAmount.toLocaleString()}`
                          : "Set a monthly limit to calculate"}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 flex-1">
                      <button
                        type="button"
                        onClick={() => handleToggleEmail(index)}
                        disabled={!emailIntegrationEnabled}
                        className="flex items-center gap-1.5 disabled:cursor-not-allowed"
                        title={
                          emailIntegrationEnabled
                            ? "Email org admins"
                            : "Email alerts require RESEND_API_KEY in deployment (restart required)"
                        }
                      >
                        <PillBadge
                          active={
                            emailIntegrationEnabled && threshold.email_enabled
                          }
                          icon={<EmailIcon />}
                          label="Email org admins"
                          disabled={!emailIntegrationEnabled}
                        />
                      </button>
                      <button
                        type="button"
                        onClick={() => handleToggleSlack(index)}
                        disabled={!slackIntegrationEnabled}
                        className="flex items-center gap-1.5 disabled:cursor-not-allowed"
                        title={
                          slackIntegrationEnabled
                            ? "Post to Slack"
                            : "Slack integration must be configured in deployment (restart required)"
                        }
                      >
                        <PillBadge
                          active={
                            slackIntegrationEnabled && threshold.slack_enabled
                          }
                          icon={<SlackIcon />}
                          label="# Post to Slack"
                          disabled={!slackIntegrationEnabled}
                        />
                      </button>
                    </div>
                    <button
                      type="button"
                      onClick={() => handleDeleteThreshold(index)}
                      className="p-1.5 text-[#6B6B6B] hover:text-red-400 hover:bg-red-500/10 rounded transition-colors"
                    >
                      <TrashIcon />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Slack Integration Footer */}
          <div className="mb-6 p-4 bg-[#0B0F17] rounded-lg border border-[#262626]">
            <label
              htmlFor="slack-channel"
              className="block text-sm text-[#8C8C8C] mb-2"
            >
              Slack channel
            </label>
            <div className="relative">
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[#6B6B6B]">
                <HashIcon />
              </span>
              <input
                id="slack-channel"
                type="text"
                value={slackChannel}
                onChange={(e) => {
                  if (!slackIntegrationEnabled) return;
                  setSlackChannel(e.target.value);
                }}
                disabled={!slackIntegrationEnabled}
                placeholder={
                  slackIntegrationEnabled
                    ? "#budget-alerts"
                    : "Connect Slack to set a channel"
                }
                className="w-full pl-9 pr-4 py-2 bg-[#151D2A] border border-[#262626] rounded-lg text-white focus:outline-none focus:border-blue-500 disabled:opacity-60 disabled:cursor-not-allowed"
              />
            </div>
            {slackIntegrationEnabled ? (
              <p className="text-xs text-[#6B6B6B] mt-2">
                Used by any threshold with &apos;Post to Slack&apos; enabled.
              </p>
            ) : (
              <p className="text-xs text-amber-400 mt-2">
                Slack alerts are disabled. Please integrate Slack to select a
                channel.
              </p>
            )}
          </div>

          {/* Action Buttons */}
          <div className="flex justify-end gap-3">
            <button
              type="button"
              onClick={handleReset}
              className="px-4 py-2 text-sm text-[#8C8C8C] bg-[#0B0F17] border border-[#262626] rounded-lg hover:bg-[#1E1E1E] transition-colors"
            >
              Reset
            </button>
            <button
              type="button"
              onClick={handleSaveOrgBudget}
              disabled={updateBudgets.isPending || !isMonthlyLimitValid}
              className="px-4 py-2 text-sm text-white bg-blue-500 rounded-lg hover:bg-blue-600 transition-colors disabled:opacity-60"
            >
              Save changes
            </button>
          </div>
        </div>
      )}

      {activeTab === "defaults" && (
        <div className="bg-[#151D2A] border border-[#262626] rounded-lg p-6">
          <div className="mb-6">
            <h2 className="text-lg font-medium text-white mb-1">
              Default budget for new users
            </h2>
            <p className="text-sm text-[#8C8C8C]">
              Applied automatically when a user joins your organization.
              Existing users keep their current budgets.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-6">
            <div>
              <div className="block text-sm text-[#8C8C8C] mb-2">
                Budget cadence
              </div>
              <div className="px-4 py-2 bg-[#0B0F17] border border-[#262626] rounded-lg text-sm text-white">
                Monthly
              </div>
            </div>
            <div>
              <label
                htmlFor="default-budget-amount"
                className="block text-sm text-[#8C8C8C] mb-2"
              >
                Default amount
              </label>
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[#6B6B6B]">
                  $
                </span>
                <input
                  id="default-budget-amount"
                  type="number"
                  value={defaultAmount}
                  onChange={(e) => setDefaultAmount(e.target.value)}
                  className="w-full pl-7 pr-4 py-2 bg-[#0B0F17] border border-[#262626] rounded-lg text-white focus:outline-none focus:border-blue-500"
                />
              </div>
            </div>
          </div>

          {/* Preview */}
          <div className="mt-6">
            <div className="block text-sm text-[#8C8C8C] mb-2">Preview</div>
            <div className="p-4 bg-[#0B0F17] rounded-lg border border-[#262626]">
              <p className="text-sm text-[#8C8C8C]">
                {`New users get up to $${defaultAmountLabel} per month before requiring an increase.`}
              </p>
            </div>
          </div>

          {/* Action Button */}
          <div className="flex justify-end mt-6">
            <button
              type="button"
              onClick={handleSaveDefault}
              disabled={updateBudgets.isPending}
              className="px-4 py-2 text-sm text-white bg-blue-500 rounded-lg hover:bg-blue-600 transition-colors disabled:opacity-60"
            >
              Save default
            </button>
          </div>
        </div>
      )}

      {activeTab === "overrides" && (
        <div className="bg-[#151D2A] border border-[#262626] rounded-lg p-6">
          <div className="flex items-start justify-between mb-6">
            <div>
              <h2 className="text-lg font-medium text-white mb-1">
                User budget overrides
              </h2>
              <p className="text-sm text-[#8C8C8C]">
                Override the default for individual users — increase, decrease,
                or disable.
              </p>
            </div>
          </div>

          {/* Search and Filter Bar */}
          <div className="flex gap-4 mb-6">
            <div className="relative flex-1">
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[#6B6B6B]">
                <SearchIcon />
              </span>
              <input
                type="text"
                placeholder="Search users by name or email..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-[#0B0F17] border border-[#262626] rounded-lg text-white placeholder-[#6B6B6B] focus:outline-none focus:border-blue-500"
              />
            </div>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-4 py-2 bg-[#0B0F17] border border-[#262626] rounded-lg text-white focus:outline-none focus:border-blue-500"
            >
              <option value="all">All statuses</option>
              <option value="over80">Over 80%</option>
              <option value="over90">Over 90%</option>
              <option value="overCap">Over cap</option>
              <option value="onTrack">On track</option>
              <option value="noCap">No cap</option>
              <option value="disabled">Disabled</option>
            </select>
          </div>

          {/* User Table */}
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[#262626]">
                  <th className="px-4 py-3 text-left text-xs font-medium text-[#6B6B6B] uppercase tracking-wider">
                    User
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-[#6B6B6B] uppercase tracking-wider">
                    Budget
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-[#6B6B6B] uppercase tracking-wider">
                    Usage
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-[#6B6B6B] uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-[#6B6B6B] uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {filteredUsers.map((user) => {
                  const isEditing = editingUserId === user.user_id;
                  const overrideValue = Number(overrideAmount);
                  const canSaveOverride =
                    overrideDisabled ||
                    (!!overrideAmount &&
                      !Number.isNaN(overrideValue) &&
                      overrideValue > 0);

                  return (
                    <tr
                      key={user.user_id}
                      className="border-b border-[#262626] hover:bg-[#1E1E1E]/50 transition-colors"
                    >
                      <td className="px-4 py-4">
                        <div className="flex items-center gap-3">
                          <Avatar name={user.name} />
                          <div>
                            <div className="text-white font-medium">
                              {user.name}
                            </div>
                            <div className="text-sm text-[#6B6B6B]">
                              {user.email || "-"}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-4">
                        {isEditing ? (
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <span className="text-[#6B6B6B]">$</span>
                              <input
                                type="number"
                                min="0"
                                step="0.01"
                                value={overrideAmount}
                                onChange={(e) =>
                                  setOverrideAmount(e.target.value)
                                }
                                disabled={overrideDisabled}
                                className="w-28 px-2 py-1 bg-[#0B0F17] border border-[#262626] rounded text-white focus:outline-none focus:border-blue-500 disabled:opacity-60"
                              />
                              <span className="text-xs text-[#6B6B6B]">
                                / month
                              </span>
                            </div>
                            <label className="flex items-center gap-2 text-xs text-[#8C8C8C]">
                              <input
                                type="checkbox"
                                checked={overrideDisabled}
                                onChange={(e) =>
                                  setOverrideDisabled(e.target.checked)
                                }
                                className="accent-blue-500"
                              />
                              Disable budget for this user
                            </label>
                          </div>
                        ) : (
                          <>
                            <div className="text-white">{user.budgetLabel}</div>
                            <div className="flex items-center gap-1.5 text-xs text-[#6B6B6B]">
                              {user.is_override && (
                                <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                              )}
                              {user.budgetNote}
                            </div>
                          </>
                        )}
                      </td>
                      <td className="px-4 py-4 min-w-[180px]">
                        {user.hasLimit ? (
                          <div>
                            <UserProgressBar
                              value={user.usage}
                              max={user.maxUsage}
                              status={user.statusColor}
                            />
                            <div className="mt-1 text-xs text-[#6B6B6B]">
                              {`$${user.usage.toLocaleString("en-US", {
                                minimumFractionDigits: 2,
                                maximumFractionDigits: 2,
                              })} of $${user.maxUsage.toLocaleString("en-US", {
                                minimumFractionDigits: 2,
                                maximumFractionDigits: 2,
                              })}`}
                            </div>
                          </div>
                        ) : (
                          <div className="text-sm text-[#8C8C8C]">
                            {`$${user.usage.toLocaleString("en-US", {
                              minimumFractionDigits: 2,
                              maximumFractionDigits: 2,
                            })} spent`}
                          </div>
                        )}
                      </td>
                      <td className="px-4 py-4">
                        <StatusPill status={user.status} />
                      </td>
                      <td className="px-4 py-4 text-right">
                        {isEditing ? (
                          <div className="flex items-center justify-end gap-2">
                            <button
                              type="button"
                              onClick={() => saveOverride(user.user_id)}
                              disabled={
                                !canSaveOverride || upsertOverride.isPending
                              }
                              className="px-3 py-1.5 text-sm text-white bg-blue-500 rounded hover:bg-blue-600 transition-colors disabled:opacity-60"
                            >
                              Save
                            </button>
                            <button
                              type="button"
                              onClick={cancelEditing}
                              className="px-3 py-1.5 text-sm text-[#8C8C8C] hover:text-white hover:bg-[#262626] rounded transition-colors"
                            >
                              Cancel
                            </button>
                          </div>
                        ) : (
                          <div className="flex items-center justify-end gap-2">
                            <button
                              type="button"
                              onClick={() => startEditingUser(user)}
                              className="px-3 py-1.5 text-sm text-[#8C8C8C] hover:text-white hover:bg-[#262626] rounded transition-colors"
                            >
                              Edit
                            </button>
                            {user.is_override && (
                              <button
                                type="button"
                                onClick={() => removeOverride(user.user_id)}
                                disabled={deleteOverride.isPending}
                                className="p-1.5 text-[#6B6B6B] hover:text-red-400 hover:bg-red-500/10 rounded transition-colors disabled:opacity-60"
                              >
                                <TrashIcon />
                              </button>
                            )}
                          </div>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {filteredUsers.length === 0 && (
            <div className="py-12 text-center text-[#6B6B6B]">
              No users found matching your criteria.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
