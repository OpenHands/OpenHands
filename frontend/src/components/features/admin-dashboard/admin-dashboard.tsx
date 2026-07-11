/* eslint-disable i18next/no-literal-string */
import React from "react";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { UsageDashboard } from "./usage-dashboard";

export function AdminDashboard() {
  const { organizationId } = useSelectedOrganizationId();

  if (!organizationId) {
    return (
      <div className="p-8 text-center text-[#8C8C8C]">
        Please select an organization to view Usage & Monitoring.
      </div>
    );
  }

  return <UsageDashboard />;
}
