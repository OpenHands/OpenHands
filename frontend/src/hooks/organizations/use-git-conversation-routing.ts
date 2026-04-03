import React from "react";
import {
  useUserGitOrganizations,
  useGitClaims,
} from "#/hooks/query/use-git-organizations";
import { useClaimGitOrg } from "#/hooks/mutation/use-claim-git-org";
import { useDisconnectGitOrg } from "#/hooks/mutation/use-disconnect-git-org";
import type { GitOrg } from "#/types/org";

function buildOrgId(provider: string, name: string): string {
  return `${provider.toLowerCase()}:${name.toLowerCase()}`;
}

export function useGitConversationRouting() {
  const { data: userGitOrgs, isLoading: isLoadingUserOrgs } =
    useUserGitOrganizations();
  const { data: claims, isLoading: isLoadingClaims } = useGitClaims();
  const { mutate: claimGitOrg } = useClaimGitOrg();
  const { mutate: disconnectGitOrg } = useDisconnectGitOrg();

  const [pendingClaims, setPendingClaims] = React.useState<Set<string>>(
    new Set(),
  );
  const [pendingDisconnects, setPendingDisconnects] = React.useState<
    Set<string>
  >(new Set());

  const orgs = React.useMemo<GitOrg[]>(() => {
    if (!userGitOrgs) return [];

    const claimMap = new Map(
      (claims ?? []).map((c) => [
        buildOrgId(c.provider, c.git_organization),
        c,
      ]),
    );

    return userGitOrgs.organizations.map((name) => {
      const id = buildOrgId(userGitOrgs.provider, name);
      const claim = claimMap.get(id);

      let status: GitOrg["status"] = "unclaimed";
      if (pendingClaims.has(id)) {
        status = "claiming";
      } else if (pendingDisconnects.has(id)) {
        status = "disconnecting";
      } else if (claim) {
        status = "claimed";
      }

      return {
        id,
        claimId: claim?.id ?? null,
        provider: userGitOrgs.provider,
        name,
        status,
      };
    });
  }, [userGitOrgs, claims, pendingClaims, pendingDisconnects]);

  const orgsRef = React.useRef(orgs);
  orgsRef.current = orgs;

  const claimOrg = React.useCallback(
    (id: string) => {
      const org = orgsRef.current.find((o) => o.id === id);
      if (!org || org.status !== "unclaimed") return;

      setPendingClaims((prev) => new Set(prev).add(id));

      claimGitOrg(
        { provider: org.provider, gitOrganization: org.name },
        {
          onSettled: () => {
            setPendingClaims((prev) => {
              const next = new Set(prev);
              next.delete(id);
              return next;
            });
          },
        },
      );
    },
    [claimGitOrg],
  );

  const disconnectOrg = React.useCallback(
    (id: string) => {
      const org = orgsRef.current.find((o) => o.id === id);
      if (!org || org.status !== "claimed" || !org.claimId) return;

      setPendingDisconnects((prev) => new Set(prev).add(id));

      disconnectGitOrg(
        { claimId: org.claimId },
        {
          onSettled: () => {
            setPendingDisconnects((prev) => {
              const next = new Set(prev);
              next.delete(id);
              return next;
            });
          },
        },
      );
    },
    [disconnectGitOrg],
  );

  const isLoading = isLoadingUserOrgs || isLoadingClaims;

  return { orgs, claimOrg, disconnectOrg, isLoading };
}
