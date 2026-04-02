import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import {
  displaySuccessToast,
  displayErrorToast,
} from "#/utils/custom-toast-handlers";

// TODO: This entire hook uses mock data and simulated async behavior.
// Replace with real API calls (e.g., organizationService.claimOrg / disconnectOrg)
// once the backend endpoints for organization claims are implemented.
export interface GitOrg {
  id: string;
  provider: "GitHub" | "GitLab";
  name: string;
  status: "unclaimed" | "claimed" | "claiming" | "disconnecting";
}

// TODO: Remove mock data once the backend API for fetching available git organizations is ready.
const INITIAL_ORGS: GitOrg[] = [
  { id: "1", provider: "GitHub", name: "OpenHands", status: "claimed" },
  { id: "2", provider: "GitHub", name: "AcmeCo", status: "unclaimed" },
  {
    id: "3",
    provider: "GitHub",
    name: "already-claimed",
    status: "unclaimed",
  },
  { id: "4", provider: "GitLab", name: "OpenHands", status: "unclaimed" },
];

export function useGitConversationRouting() {
  const { t } = useTranslation();
  const [orgs, setOrgs] = React.useState<GitOrg[]>(INITIAL_ORGS);

  const updateOrgStatus = React.useCallback(
    (id: string, status: GitOrg["status"]) => {
      setOrgs((prev) =>
        prev.map((org) => (org.id === id ? { ...org, status } : org)),
      );
    },
    [],
  );

  const claimOrg = React.useCallback(
    (id: string) => {
      const org = orgs.find((o) => o.id === id);
      if (!org || org.status !== "unclaimed") return;

      updateOrgStatus(id, "claiming");

      setTimeout(() => {
        if (org.name === "already-claimed") {
          updateOrgStatus(id, "unclaimed");
          displayErrorToast(t(I18nKey.ORG$CLAIM_ERROR));
        } else {
          updateOrgStatus(id, "claimed");
          displaySuccessToast(t(I18nKey.ORG$CLAIM_SUCCESS));
        }
      }, 1000);
    },
    [orgs, updateOrgStatus, t],
  );

  const disconnectOrg = React.useCallback(
    (id: string) => {
      const org = orgs.find((o) => o.id === id);
      if (!org || org.status !== "claimed") return;

      updateOrgStatus(id, "disconnecting");

      setTimeout(() => {
        updateOrgStatus(id, "unclaimed");
        displaySuccessToast(t(I18nKey.ORG$DISCONNECT_SUCCESS));
      }, 1000);
    },
    [orgs, updateOrgStatus, t],
  );

  return { orgs, claimOrg, disconnectOrg };
}
