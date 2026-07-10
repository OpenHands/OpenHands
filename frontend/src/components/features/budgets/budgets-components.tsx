/* eslint-disable i18next/no-literal-string */
import React from "react";

export function Toggle({
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

export function PillBadge({
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

export function SpendMeter({
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
          <div className="absolute top-0 left-[80%] w-px h-2 bg-[#6B6B6B]" />
          <div className="absolute top-0 left-[90%] w-px h-2 bg-[#6B6B6B]" />
          <div className="absolute top-0 left-[100%] w-px h-2 bg-[#6B6B6B]" />
        </div>
      )}
    </div>
  );
}

export function UserProgressBar({
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

export function Avatar({
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

export function StatusPill({ status }: { status: string }) {
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
