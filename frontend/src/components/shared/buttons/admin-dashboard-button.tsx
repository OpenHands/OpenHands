import React from "react";
import { useLocation } from "react-router";
import { useMe } from "#/hooks/query/use-me";
import { cn } from "#/utils/utils";

export function AdminDashboardButton() {
  const location = useLocation();
  const { data: me } = useMe();

  const isAdmin = me?.role === "owner" || me?.role === "admin";

  if (!isAdmin) {
    return null;
  }

  const isActive = location.pathname.startsWith("/admin");

  return (
    <a
      href="/admin/dashboard"
      className={cn(
        "flex items-center justify-center w-[34px] h-[34px] rounded-lg transition-colors",
        isActive
          ? "bg-[#262626] text-white"
          : "text-[#8C8C8C] hover:text-white hover:bg-[#1E1E1E]",
      )}
      title="Admin Dashboard"
      aria-label="Admin Dashboard"
    >
      <svg
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <rect x="3" y="3" width="7" height="7" />
        <rect x="14" y="3" width="7" height="7" />
        <rect x="14" y="14" width="7" height="7" />
        <rect x="3" y="14" width="7" height="7" />
      </svg>
    </a>
  );
}
