/* eslint-disable i18next/no-literal-string */
import React, { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { useConfig } from "#/hooks/query/use-config";
import { useDebounce } from "#/hooks/use-debounce";
import { BUDGET_TABS, BudgetTab, USERS_PER_PAGE } from "./budgets-constants";
import {
  DefaultBudgetsTab,
  OrganizationBudgetTab,
  UserOverridesTab,
} from "./budgets-tabs";
import type { BudgetThreshold, BudgetUserRow } from "./budgets-tabs";

export function Budgets() {
  const { organizationId } = useSelectedOrganizationId();
  const queryClient = useQueryClient();

  const { data: config } = useConfig();
  const slackIntegrationEnabled = Boolean(config?.slack_enabled);
  const [usersPage, setUsersPage] = useState(1);

  const emailIntegrationEnabled = Boolean(config?.email_enabled);

  const [activeTab, setActiveTab] = useState<BudgetTab>("organization");

  const [searchQuery, setSearchQuery] = useState("");
  const debouncedSearchQuery = useDebounce(searchQuery, 300);
  const [statusFilter, setStatusFilter] = useState("all");
  const usersSearch = debouncedSearchQuery.trim();
  const usersStatus = statusFilter === "all" ? undefined : statusFilter;

  const { data: budgetData, isLoading } = useQuery({
    queryKey: [
      "organizations",
      "budgets",
      organizationId,
      usersPage,
      usersSearch,
      usersStatus,
    ],
    queryFn: () =>
      organizationService.getBudgetSettings({
        orgId: organizationId!,
        usersPage,
        usersPerPage: USERS_PER_PAGE,
        usersSearch: usersSearch || undefined,
        usersStatus,
      }),
    enabled: !!organizationId,
  });

  useEffect(() => {
    setUsersPage(1);
  }, [organizationId]);

  useEffect(() => {
    setUsersPage(1);
  }, [debouncedSearchQuery, statusFilter]);

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
  const [thresholds, setThresholds] = useState<BudgetThreshold[]>([]);
  const [defaultAmount, setDefaultAmount] = useState("");
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
        // Cycle boundaries are UTC midnights; local rendering shifts the
        // month for viewers west of UTC (July cycle labeled "June").
        timeZone: "UTC",
      })
    : "this cycle";
  const defaultUserLimit = budgetData?.default_user_monthly_limit ?? null;

  const usersTotal = budgetData?.users_total ?? 0;
  const usersPerPage = budgetData?.users_per_page ?? USERS_PER_PAGE;
  const totalPages = Math.max(1, Math.ceil(usersTotal / usersPerPage));
  const usersStart = usersTotal === 0 ? 0 : (usersPage - 1) * usersPerPage + 1;
  const usersEnd =
    usersTotal === 0 ? 0 : usersStart + (budgetData?.users.length ?? 0) - 1;

  useEffect(() => {
    if (usersPage > totalPages) {
      setUsersPage(totalPages);
    }
  }, [totalPages, usersPage]);

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

  const startEditingUser = (user: BudgetUserRow) => {
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
        <OrganizationBudgetTab
          orgBudgetEnabled={orgBudgetEnabled}
          onToggleOrgBudget={setOrgBudgetEnabled}
          currentSpend={currentSpend}
          monthlyLimitValue={monthlyLimitValue}
          cycleLabel={cycleLabel}
          percentage={percentage}
          monthlyLimit={monthlyLimit}
          onMonthlyLimitChange={setMonthlyLimit}
          billingCycle={billingCycle}
          onBillingCycleChange={setBillingCycle}
          thresholds={thresholds}
          onAddThreshold={handleAddThreshold}
          onDeleteThreshold={handleDeleteThreshold}
          onToggleEmail={handleToggleEmail}
          onToggleSlack={handleToggleSlack}
          emailIntegrationEnabled={emailIntegrationEnabled}
          slackIntegrationEnabled={slackIntegrationEnabled}
          slackChannel={slackChannel}
          onSlackChannelChange={setSlackChannel}
          onReset={handleReset}
          onSave={handleSaveOrgBudget}
          isSaving={updateBudgets.isPending}
          isMonthlyLimitValid={isMonthlyLimitValid}
        />
      )}

      {activeTab === "defaults" && (
        <DefaultBudgetsTab
          defaultAmount={defaultAmount}
          defaultAmountLabel={defaultAmountLabel}
          onDefaultAmountChange={setDefaultAmount}
          onSave={handleSaveDefault}
          isSaving={updateBudgets.isPending}
        />
      )}

      {activeTab === "overrides" && (
        <UserOverridesTab
          searchQuery={searchQuery}
          statusFilter={statusFilter}
          onSearchChange={setSearchQuery}
          onStatusFilterChange={setStatusFilter}
          userRows={userRows}
          editingUserId={editingUserId}
          overrideAmount={overrideAmount}
          overrideDisabled={overrideDisabled}
          onOverrideAmountChange={setOverrideAmount}
          onOverrideDisabledChange={setOverrideDisabled}
          onStartEditing={startEditingUser}
          onCancelEditing={cancelEditing}
          onSaveOverride={saveOverride}
          onRemoveOverride={removeOverride}
          isSavingOverride={upsertOverride.isPending}
          isDeletingOverride={deleteOverride.isPending}
          usersTotal={usersTotal}
          usersStart={usersStart}
          usersEnd={usersEnd}
          usersPage={usersPage}
          totalPages={totalPages}
          isLoading={isLoading}
          onPageChange={setUsersPage}
        />
      )}
    </div>
  );
}
