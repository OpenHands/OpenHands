/* eslint-disable i18next/no-literal-string */
import React from "react";
import { Link } from "react-router";
import {
  EmailIcon,
  HashIcon,
  SearchIcon,
  SlackIcon,
  TrashIcon,
} from "#/components/shared/icons/inline-icons";
import {
  Avatar,
  PillBadge,
  SpendMeter,
  StatusPill,
  Toggle,
  UserProgressBar,
} from "./budgets-components";

export type BudgetThreshold = {
  percentage: number;
  email_enabled: boolean;
  slack_enabled: boolean;
};

export type BudgetUserRow = {
  user_id: string;
  name: string;
  email: string;
  is_override: boolean;
  is_disabled: boolean;
  effective_monthly_limit?: number | null;
  budgetLabel: string;
  budgetNote: string;
  hasLimit: boolean;
  usage: number;
  maxUsage: number;
  status: string;
  statusColor: "green" | "yellow" | "red";
};

interface OrganizationBudgetTabProps {
  orgBudgetEnabled: boolean;
  onToggleOrgBudget: (value: boolean) => void;
  currentSpend: number;
  monthlyLimitValue: number | null;
  cycleLabel: string;
  percentage: number;
  monthlyLimit: string;
  onMonthlyLimitChange: (value: string) => void;
  billingCycle: string;
  onBillingCycleChange: (value: string) => void;
  thresholds: BudgetThreshold[];
  onAddThreshold: () => void;
  onDeleteThreshold: (index: number) => void;
  onToggleEmail: (index: number) => void;
  onToggleSlack: (index: number) => void;
  emailIntegrationEnabled: boolean;
  slackIntegrationEnabled: boolean;
  slackChannel: string;
  onSlackChannelChange: (value: string) => void;
  onReset: () => void;
  onSave: () => void;
  isSaving: boolean;
  isMonthlyLimitValid: boolean;
}

export function OrganizationBudgetTab({
  orgBudgetEnabled,
  onToggleOrgBudget,
  currentSpend,
  monthlyLimitValue,
  cycleLabel,
  percentage,
  monthlyLimit,
  onMonthlyLimitChange,
  billingCycle,
  onBillingCycleChange,
  thresholds,
  onAddThreshold,
  onDeleteThreshold,
  onToggleEmail,
  onToggleSlack,
  emailIntegrationEnabled,
  slackIntegrationEnabled,
  slackChannel,
  onSlackChannelChange,
  onReset,
  onSave,
  isSaving,
  isMonthlyLimitValid,
}: OrganizationBudgetTabProps) {
  return (
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
            onChange={onToggleOrgBudget}
            label="Enable organization budget"
          />
        </div>
      </div>

      <div className="mb-6">
        <div className="flex items-baseline justify-between mb-3">
          <div>
            <span className="text-3xl font-bold text-white">
              {`$${currentSpend.toLocaleString("en-US", {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
              })}`}
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
              onChange={(event) => onMonthlyLimitChange(event.target.value)}
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
            onChange={(event) => onBillingCycleChange(event.target.value)}
            className="w-full px-4 py-2 bg-[#0B0F17] border border-[#262626] rounded-lg text-white focus:outline-none focus:border-blue-500"
          >
            <option value="1st">1st of each month</option>
            <option value="15th">15th of each month</option>
          </select>
        </div>
      </div>

      <div className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="text-sm font-medium text-white mb-1">
              Alert thresholds
            </h3>
            <p className="text-xs text-[#6B6B6B]">
              Add one or more thresholds. Each can email admins, post to Slack,
              or both.
            </p>
            {(!emailIntegrationEnabled || !slackIntegrationEnabled) && (
              <div className="mt-2 space-y-1 text-xs text-amber-400">
                {!emailIntegrationEnabled && (
                  <p>
                    Email alerts require RESEND_API_KEY or SMTP_* env vars set
                    in the deployment environment and a restart.
                  </p>
                )}
                {!slackIntegrationEnabled && (
                  <p>
                    Slack alerts require the Slack app to be configured in the
                    deployment (SLACK_* env vars). After a restart, connect it
                    in{" "}
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
            onClick={onAddThreshold}
            className="px-3 py-1.5 text-sm text-blue-400 border border-blue-400/30 rounded-lg hover:bg-blue-500/10 transition-colors"
          >
            + Add threshold
          </button>
        </div>

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
                    onClick={() => onToggleEmail(index)}
                    disabled={!emailIntegrationEnabled}
                    className="flex items-center gap-1.5 disabled:cursor-not-allowed"
                    title={
                      emailIntegrationEnabled
                        ? "Email org admins"
                        : "Email alerts require RESEND_API_KEY or SMTP_* env vars in deployment (restart required)"
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
                    onClick={() => onToggleSlack(index)}
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
                  onClick={() => onDeleteThreshold(index)}
                  aria-label={`Delete ${threshold.percentage}% threshold`}
                  className="p-1.5 text-[#6B6B6B] hover:text-red-400 hover:bg-red-500/10 rounded transition-colors"
                >
                  <TrashIcon />
                </button>
              </div>
            );
          })}
        </div>
      </div>

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
            onChange={(event) => {
              if (!slackIntegrationEnabled) return;
              onSlackChannelChange(event.target.value);
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

      <div className="flex justify-end gap-3">
        <button
          type="button"
          onClick={onReset}
          className="px-4 py-2 text-sm text-[#8C8C8C] bg-[#0B0F17] border border-[#262626] rounded-lg hover:bg-[#1E1E1E] transition-colors"
        >
          Reset
        </button>
        <button
          type="button"
          onClick={onSave}
          disabled={isSaving || !isMonthlyLimitValid}
          className="px-4 py-2 text-sm text-white bg-blue-500 rounded-lg hover:bg-blue-600 transition-colors disabled:opacity-60"
        >
          Save changes
        </button>
      </div>
    </div>
  );
}

interface DefaultBudgetsTabProps {
  defaultAmount: string;
  defaultAmountLabel: string;
  onDefaultAmountChange: (value: string) => void;
  onSave: () => void;
  isSaving: boolean;
}

export function DefaultBudgetsTab({
  defaultAmount,
  defaultAmountLabel,
  onDefaultAmountChange,
  onSave,
  isSaving,
}: DefaultBudgetsTabProps) {
  return (
    <div className="bg-[#151D2A] border border-[#262626] rounded-lg p-6">
      <div className="mb-6">
        <h2 className="text-lg font-medium text-white mb-1">
          Default budget for new users
        </h2>
        <p className="text-sm text-[#8C8C8C]">
          Applied automatically when a user joins your organization. Existing
          users keep their current budgets.
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
              onChange={(event) => onDefaultAmountChange(event.target.value)}
              className="w-full pl-7 pr-4 py-2 bg-[#0B0F17] border border-[#262626] rounded-lg text-white focus:outline-none focus:border-blue-500"
            />
          </div>
        </div>
      </div>

      <div className="mt-6">
        <div className="block text-sm text-[#8C8C8C] mb-2">Preview</div>
        <div className="p-4 bg-[#0B0F17] rounded-lg border border-[#262626]">
          <p className="text-sm text-[#8C8C8C]">
            {`New users get up to $${defaultAmountLabel} per month before requiring an increase.`}
          </p>
        </div>
      </div>

      <div className="flex justify-end mt-6">
        <button
          type="button"
          onClick={onSave}
          disabled={isSaving}
          className="px-4 py-2 text-sm text-white bg-blue-500 rounded-lg hover:bg-blue-600 transition-colors disabled:opacity-60"
        >
          Save default
        </button>
      </div>
    </div>
  );
}

interface UserOverridesTabProps {
  searchQuery: string;
  statusFilter: string;
  onSearchChange: (value: string) => void;
  onStatusFilterChange: (value: string) => void;
  userRows: BudgetUserRow[];
  editingUserId: string | null;
  overrideAmount: string;
  overrideDisabled: boolean;
  onOverrideAmountChange: (value: string) => void;
  onOverrideDisabledChange: (value: boolean) => void;
  onStartEditing: (user: BudgetUserRow) => void;
  onCancelEditing: () => void;
  onSaveOverride: (userId: string) => void;
  onRemoveOverride: (userId: string) => void;
  isSavingOverride: boolean;
  isDeletingOverride: boolean;
  usersTotal: number;
  usersStart: number;
  usersEnd: number;
  usersPage: number;
  totalPages: number;
  isLoading: boolean;
  onPageChange: (page: number) => void;
}

export function UserOverridesTab({
  searchQuery,
  statusFilter,
  onSearchChange,
  onStatusFilterChange,
  userRows,
  editingUserId,
  overrideAmount,
  overrideDisabled,
  onOverrideAmountChange,
  onOverrideDisabledChange,
  onStartEditing,
  onCancelEditing,
  onSaveOverride,
  onRemoveOverride,
  isSavingOverride,
  isDeletingOverride,
  usersTotal,
  usersStart,
  usersEnd,
  usersPage,
  totalPages,
  isLoading,
  onPageChange,
}: UserOverridesTabProps) {
  return (
    <div className="bg-[#151D2A] border border-[#262626] rounded-lg p-6">
      <div className="flex items-start justify-between mb-6">
        <div>
          <h2 className="text-lg font-medium text-white mb-1">
            User budget overrides
          </h2>
          <p className="text-sm text-[#8C8C8C]">
            Override the default for individual users — increase, decrease, or
            disable.
          </p>
        </div>
      </div>

      <div className="flex gap-4 mb-6">
        <div className="relative flex-1">
          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[#6B6B6B]">
            <SearchIcon />
          </span>
          <input
            type="text"
            placeholder="Search users by name or email..."
            value={searchQuery}
            onChange={(event) => onSearchChange(event.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-[#0B0F17] border border-[#262626] rounded-lg text-white placeholder-[#6B6B6B] focus:outline-none focus:border-blue-500"
          />
        </div>
        <select
          value={statusFilter}
          onChange={(event) => onStatusFilterChange(event.target.value)}
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
            {userRows.map((user) => {
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
                            onChange={(event) =>
                              onOverrideAmountChange(event.target.value)
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
                            onChange={(event) =>
                              onOverrideDisabledChange(event.target.checked)
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
                          onClick={() => onSaveOverride(user.user_id)}
                          disabled={!canSaveOverride || isSavingOverride}
                          className="px-3 py-1.5 text-sm text-white bg-blue-500 rounded hover:bg-blue-600 transition-colors disabled:opacity-60"
                        >
                          Save
                        </button>
                        <button
                          type="button"
                          onClick={onCancelEditing}
                          className="px-3 py-1.5 text-sm text-[#8C8C8C] hover:text-white hover:bg-[#262626] rounded transition-colors"
                        >
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <div className="flex items-center justify-end gap-2">
                        <button
                          type="button"
                          onClick={() => onStartEditing(user)}
                          className="px-3 py-1.5 text-sm text-[#8C8C8C] hover:text-white hover:bg-[#262626] rounded transition-colors"
                        >
                          Edit
                        </button>
                        {user.is_override && (
                          <button
                            type="button"
                            onClick={() => onRemoveOverride(user.user_id)}
                            disabled={isDeletingOverride}
                            aria-label={`Remove override for ${user.name}`}
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

      {usersTotal > 0 && (
        <div className="mt-4 flex flex-col gap-3 text-sm text-[#6B6B6B] sm:flex-row sm:items-center sm:justify-between">
          <span>{`Showing ${usersStart}-${usersEnd} of ${usersTotal}`}</span>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => onPageChange(Math.max(1, usersPage - 1))}
              disabled={usersPage <= 1 || isLoading}
              className="px-3 py-1.5 text-sm text-[#8C8C8C] hover:text-white hover:bg-[#262626] rounded transition-colors disabled:opacity-60"
            >
              Previous
            </button>
            <span className="text-xs text-[#6B6B6B]">
              {`${usersPage} / ${totalPages}`}
            </span>
            <button
              type="button"
              onClick={() => onPageChange(Math.min(totalPages, usersPage + 1))}
              disabled={usersPage >= totalPages || isLoading}
              className="px-3 py-1.5 text-sm text-[#8C8C8C] hover:text-white hover:bg-[#262626] rounded transition-colors disabled:opacity-60"
            >
              Next
            </button>
          </div>
        </div>
      )}

      {userRows.length === 0 && (
        <div className="py-12 text-center text-[#6B6B6B]">
          No users found matching your criteria.
        </div>
      )}
    </div>
  );
}
