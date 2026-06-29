/* eslint-disable i18next/no-literal-string */
import React, { useState } from "react";

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
}: {
  enabled: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={enabled}
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
}: {
  active: boolean;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border transition-colors cursor-pointer ${
        active
          ? "bg-blue-500/10 text-blue-400 border-blue-500/30"
          : "bg-[#151D2A] text-[#6B6B6B] border-[#262626]"
      }`}
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
    if (percentage >= 90) return "bg-gradient-to-r from-green-500 via-yellow-500 to-red-500";
    if (percentage >= 80) return "bg-gradient-to-r from-green-500 via-yellow-500 to-orange-500";
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
  const percentage = (value / max) * 100;
  const colorClass =
    status === "red"
      ? "bg-red-500"
      : status === "yellow"
        ? "bg-yellow-500"
        : "bg-green-500";

  return (
    <div className="w-full">
      <div className="relative w-full h-1.5 bg-[#0B0F17] rounded-full overflow-hidden">
        <div
          className={`absolute inset-y-0 left-0 rounded-full ${colorClass}`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
      <div className="text-xs text-[#8C8C8C] mt-1">${value.toLocaleString()} / ${max.toLocaleString()}</div>
    </div>
  );
}

// Avatar component
function Avatar({
  name,
  size = "md",
}: {
  name: string;
  size?: "sm" | "md";
}) {
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

// Sample data
const SAMPLE_THRESHOLDS = [
  {
    percentage: 80,
    amount: 8000,
    emailActive: true,
    slackActive: false,
  },
  {
    percentage: 90,
    amount: 9000,
    emailActive: true,
    slackActive: true,
  },
  {
    percentage: 100,
    amount: 10000,
    emailActive: true,
    slackActive: true,
  },
];

const SAMPLE_USERS = [
  {
    name: "Alice Chen",
    email: "alice@acme.com",
    budget: "$500 lifetime",
    budgetNote: "Inherits default",
    isOverride: false,
    usage: 412,
    maxUsage: 500,
    status: "> 80% used",
    statusColor: "yellow" as const,
  },
  {
    name: "Bob Martinez",
    email: "bob@acme.com",
    budget: "$1,000 monthly",
    budgetNote: "Override",
    isOverride: true,
    usage: 952,
    maxUsage: 1000,
    status: "> 90% used",
    statusColor: "red" as const,
  },
  {
    name: "Chen Wei",
    email: "chen@acme.com",
    budget: "$500 lifetime",
    budgetNote: "Inherits default",
    isOverride: false,
    usage: 88,
    maxUsage: 500,
    status: "Inherits default",
    statusColor: "green" as const,
  },
  {
    name: "Dana Park",
    email: "dana@acme.com",
    budget: "$2,000 monthly",
    budgetNote: "Override",
    isOverride: true,
    usage: 1611,
    maxUsage: 2000,
    status: "> 80% used",
    statusColor: "yellow" as const,
  },
  {
    name: "Eli Brooks",
    email: "eli@acme.com",
    budget: "$500 lifetime",
    budgetNote: "Inherits default",
    isOverride: false,
    usage: 510,
    maxUsage: 500,
    status: "Over cap",
    statusColor: "red" as const,
  },
  {
    name: "Farah Idris",
    email: "farah@acme.com",
    budget: "$750 monthly",
    budgetNote: "Override",
    isOverride: true,
    usage: 220,
    maxUsage: 750,
    status: "On track",
    statusColor: "green" as const,
  },
  {
    name: "Greg Thompson",
    email: "greg@acme.com",
    budget: "$500 lifetime",
    budgetNote: "Inherits default",
    isOverride: false,
    usage: 460,
    maxUsage: 500,
    status: "> 90% used",
    statusColor: "red" as const,
  },
];

export function Budgets() {
  const [orgBudgetEnabled, setOrgBudgetEnabled] = useState(true);
  const [monthlyLimit, setMonthlyLimit] = useState("10000");
  const [billingCycle, setBillingCycle] = useState("1st");
  const [slackChannel, setSlackChannel] = useState("finance-alerts");
  const [thresholds, setThresholds] = useState(SAMPLE_THRESHOLDS);
  const [budgetType, setBudgetType] = useState<"lifetime" | "monthly">("lifetime");
  const [defaultAmount, setDefaultAmount] = useState("500");
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [users] = useState(SAMPLE_USERS);

  const currentSpend = 8213.42;
  const percentage = (currentSpend / parseFloat(monthlyLimit || "1")) * 100;

  const handleDeleteThreshold = (index: number) => {
    setThresholds(thresholds.filter((_, i) => i !== index));
  };

  const handleToggleEmail = (index: number) => {
    setThresholds(
      thresholds.map((t, i) =>
        i === index ? { ...t, emailActive: !t.emailActive } : t,
      ),
    );
  };

  const handleToggleSlack = (index: number) => {
    setThresholds(
      thresholds.map((t, i) =>
        i === index ? { ...t, slackActive: !t.slackActive } : t,
      ),
    );
  };

  const filteredUsers = users.filter((user) => {
    const matchesSearch =
      user.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      user.email.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus =
      statusFilter === "all" ||
      (statusFilter === "over80" &&
        (user.status.includes("> 80%") || user.status.includes("Over cap"))) ||
      (statusFilter === "onTrack" && user.status.includes("On track")) ||
      (statusFilter === "inherited" && user.status.includes("Inherits"));
    return matchesSearch && matchesStatus;
  });

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-semibold text-white mb-1">Budgets</h1>
        <p className="text-[#8C8C8C]">
          Control your AI spend at the organization and user level.
        </p>
      </div>

      {/* Section 1: Organization Monthly Budget */}
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
            />
          </div>
        </div>

        {/* Spend Meter */}
        <div className="mb-6">
          <div className="flex items-baseline justify-between mb-3">
            <div>
              <span className="text-3xl font-bold text-white">
                ${currentSpend.toLocaleString("en-US", {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
              </span>
              <span className="text-[#8C8C8C] ml-2">
                of ${parseFloat(monthlyLimit).toLocaleString()} spent in June
              </span>
            </div>
            <span className="text-xl font-semibold text-yellow-400">
              {percentage.toFixed(1)}%
            </span>
          </div>
          <SpendMeter percentage={percentage} />
        </div>

        {/* Inputs */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div>
            <label className="block text-sm text-[#8C8C8C] mb-2">
              Monthly limit
            </label>
            <div className="relative">
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[#6B6B6B]">
                $
              </span>
              <input
                type="number"
                value={monthlyLimit}
                onChange={(e) => setMonthlyLimit(e.target.value)}
                className="w-full pl-7 pr-4 py-2 bg-[#0B0F17] border border-[#262626] rounded-lg text-white focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>
          <div>
            <label className="block text-sm text-[#8C8C8C] mb-2">
              Billing cycle resets
            </label>
            <select
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
            </div>
            <button
              type="button"
              className="px-3 py-1.5 text-sm text-blue-400 border border-blue-400/30 rounded-lg hover:bg-blue-500/10 transition-colors"
            >
              + Add threshold
            </button>
          </div>

          {/* Threshold rows */}
          <div className="space-y-3">
            {thresholds.map((threshold, index) => (
              <div
                key={index}
                className="flex items-center gap-4 p-3 bg-[#0B0F17] rounded-lg border border-[#262626]"
              >
                <div className="w-16">
                  <span className="text-white font-medium">
                    {threshold.percentage}%
                  </span>
                </div>
                <div className="w-28">
                  <span className="text-[#8C8C8C] text-sm">
                    Triggers at ${threshold.amount.toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center gap-2 flex-1">
                  <button
                    type="button"
                    onClick={() => handleToggleEmail(index)}
                    className="flex items-center gap-1.5"
                  >
                    <PillBadge
                      active={threshold.emailActive}
                      icon={<EmailIcon />}
                      label="Email org admins"
                    />
                  </button>
                  <button
                    type="button"
                    onClick={() => handleToggleSlack(index)}
                    className="flex items-center gap-1.5"
                  >
                    <PillBadge
                      active={threshold.slackActive}
                      icon={<SlackIcon />}
                      label="# Post to Slack"
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
            ))}
          </div>
        </div>

        {/* Slack Integration Footer */}
        <div className="mb-6 p-4 bg-[#0B0F17] rounded-lg border border-[#262626]">
          <label className="block text-sm text-[#8C8C8C] mb-2">
            Slack channel
          </label>
          <div className="relative">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[#6B6B6B]">
              <HashIcon />
            </span>
            <input
              type="text"
              value={slackChannel}
              onChange={(e) => setSlackChannel(e.target.value)}
              className="w-full pl-9 pr-4 py-2 bg-[#151D2A] border border-[#262626] rounded-lg text-white focus:outline-none focus:border-blue-500"
            />
          </div>
          <p className="text-xs text-[#6B6B6B] mt-2">
            Used by any threshold with &apos;Post to Slack&apos; enabled.
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex justify-end gap-3">
          <button
            type="button"
            className="px-4 py-2 text-sm text-[#8C8C8C] bg-[#0B0F17] border border-[#262626] rounded-lg hover:bg-[#1E1E1E] transition-colors"
          >
            Reset
          </button>
          <button
            type="button"
            className="px-4 py-2 text-sm text-white bg-blue-500 rounded-lg hover:bg-blue-600 transition-colors"
          >
            Save changes
          </button>
        </div>
      </div>

      {/* Section 2: Default Budget for New Users */}
      <div className="bg-[#151D2A] border border-[#262626] rounded-lg p-6">
        <div className="mb-6">
          <h2 className="text-lg font-medium text-white mb-1">
            Default budget for new users
          </h2>
          <p className="text-sm text-[#8C8C8C]">
            Applied automatically when a user joins Acme Inc. Existing users
            keep their current budgets.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div>
            <label className="block text-sm text-[#8C8C8C] mb-2">
              Budget type
            </label>
            <div className="flex bg-[#0B0F17] border border-[#262626] rounded-lg p-1">
              <button
                type="button"
                onClick={() => setBudgetType("lifetime")}
                className={`flex-1 px-4 py-2 text-sm rounded-md transition-colors ${
                  budgetType === "lifetime"
                    ? "bg-[#262626] text-white"
                    : "text-[#6B6B6B] hover:text-white"
                }`}
              >
                Lifetime
              </button>
              <button
                type="button"
                onClick={() => setBudgetType("monthly")}
                className={`flex-1 px-4 py-2 text-sm rounded-md transition-colors ${
                  budgetType === "monthly"
                    ? "bg-[#262626] text-white"
                    : "text-[#6B6B6B] hover:text-white"
                }`}
              >
                Monthly
              </button>
            </div>
          </div>
          <div>
            <label className="block text-sm text-[#8C8C8C] mb-2">
              Default amount
            </label>
            <div className="relative">
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[#6B6B6B]">
                $
              </span>
              <input
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
          <label className="block text-sm text-[#8C8C8C] mb-2">Preview</label>
          <div className="p-4 bg-[#0B0F17] rounded-lg border border-[#262626]">
            <p className="text-sm text-[#8C8C8C]">
              New users get up to $
              {parseFloat(defaultAmount).toLocaleString()}{" "}
              {budgetType === "lifetime" ? "total" : "per month"} before
              requiring an increase.
            </p>
          </div>
        </div>

        {/* Action Button */}
        <div className="flex justify-end mt-6">
          <button
            type="button"
            className="px-4 py-2 text-sm text-white bg-blue-500 rounded-lg hover:bg-blue-600 transition-colors"
          >
            Save default
          </button>
        </div>
      </div>

      {/* Section 3: User Budget Overrides */}
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
          <button
            type="button"
            className="px-3 py-1.5 text-sm text-blue-400 border border-blue-400/30 rounded-lg hover:bg-blue-500/10 transition-colors"
          >
            + Add override
          </button>
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
            <option value="onTrack">On track</option>
            <option value="inherited">Inherits default</option>
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
              {filteredUsers.map((user, index) => (
                <tr
                  key={index}
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
                          {user.email}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-4">
                    <div className="text-white">{user.budget}</div>
                    <div className="flex items-center gap-1.5 text-xs text-[#6B6B6B]">
                      {user.isOverride && (
                        <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                      )}
                      {user.budgetNote}
                    </div>
                  </td>
                  <td className="px-4 py-4 min-w-[180px]">
                    <UserProgressBar
                      value={user.usage}
                      max={user.maxUsage}
                      status={user.statusColor}
                    />
                  </td>
                  <td className="px-4 py-4">
                    <StatusPill status={user.status} />
                  </td>
                  <td className="px-4 py-4 text-right">
                    <button
                      type="button"
                      className="px-3 py-1.5 text-sm text-[#8C8C8C] hover:text-white hover:bg-[#262626] rounded transition-colors"
                    >
                      Edit
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {filteredUsers.length === 0 && (
          <div className="py-12 text-center text-[#6B6B6B]">
            No users found matching your criteria.
          </div>
        )}
      </div>
    </div>
  );
}
