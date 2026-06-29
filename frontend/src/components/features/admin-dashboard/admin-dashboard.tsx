/* eslint-disable i18next/no-literal-string */
import React from "react";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { useOrganizations } from "#/hooks/query/use-organizations";
import { UsageDashboard } from "./usage-dashboard";

export function AdminDashboard() {
  const { organizationId } = useSelectedOrganizationId();
  const { data: orgData } = useOrganizations();

  const currentOrg = orgData?.organizations?.find(
    (org) => org.id === organizationId,
  );

  if (!organizationId) {
    return (
      <div className="p-8 text-center text-[#8C8C8C]">
        Please select an organization to view Usage & Monitoring.
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0B0B0B] text-white flex flex-col">
      <header className="border-b border-white/10 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-white font-bold text-lg">Usage & Monitoring</h1>
            <p className="text-[#888888] text-sm mt-1">
              {currentOrg?.name || "your organization"}
            </p>
          </div>
        </div>
      </header>
      <main className="flex-1 overflow-auto">
        <UsageDashboard />
      </main>
    </div>
  );
}
