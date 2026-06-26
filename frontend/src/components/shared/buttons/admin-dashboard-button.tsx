import React from "react";
import { Link, useLocation } from "react-router";
import { useConfig } from "#/hooks/query/use-config";
import { useMe } from "#/hooks/query/use-me";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { cn } from "#/utils/utils";

export function AdminDashboardButton() {
  const location = useLocation();
  const { data: config } = useConfig();
  const { organizationId } = useSelectedOrganizationId();

  const isSaas = config?.app_mode === "saas";

  const { data: meData, isLoading } = useMe();

  const isAdmin = meData?.role === "owner" || meData?.role === "admin";

  // Don't render anything while loading (prevents flash of wrong state)
  if (!isSaas || isLoading || !organizationId) {
    return null;
  }

  if (!isAdmin) {
    return null;
  }

  const isActive = location.pathname.startsWith("/settings/admin-dashboard");

  return (
    <Link
      to="/settings/admin-dashboard"
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
    </Link>
  );
}
