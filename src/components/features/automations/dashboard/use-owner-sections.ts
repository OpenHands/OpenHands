import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useCloudOrgMembers } from "#/hooks/query/use-cloud-org-member";
import { useServiceAccounts } from "#/hooks/query/use-service-accounts";
import { useAutomationPermissions } from "#/hooks/use-automation-permissions";
import type { Automation } from "#/types/automation";
import { groupAutomationsByOwner } from "../automation-group-by";

interface OwnerSection {
  key: string;
  title: string;
  automations: Automation[];
}

/**
 * The automations as one section per identity they run as. The caller's own
 * section comes first, then the other people, then service accounts, each
 * alphabetical by the name shown, and automations with no known owner last.
 * Member lookups run only while `enabled`.
 */
export function useOwnerSections(
  automations: readonly Automation[],
  { enabled }: { enabled: boolean },
): OwnerSection[] {
  const { t } = useTranslation("openhands");
  const { userId: currentUserId } = useAutomationPermissions();
  const serviceAccounts = useServiceAccounts().data;

  const groups = useMemo(
    () => (enabled ? groupAutomationsByOwner(automations) : []),
    [enabled, automations],
  );
  const memberIds = useMemo(
    () =>
      groups.flatMap((group) =>
        group.ownerId && !group.isServiceAccount ? [group.ownerId] : [],
      ),
    [groups],
  );
  const members = useCloudOrgMembers(memberIds, { enabled });

  return useMemo(() => {
    const sections = groups.map((group) => {
      const { ownerId } = group;
      let title = t(I18nKey.AUTOMATIONS$NO_OWNER);
      if (ownerId && group.isServiceAccount) {
        title =
          serviceAccounts?.find((account) => account.id === ownerId)?.name ??
          t(I18nKey.AUTOMATIONS$SERVICE_ACCOUNT_LABEL);
      } else if (ownerId) {
        const member = members.get(ownerId);
        // A member who left the org, or an app server without the member
        // lookup, has no email, so the id stands in once the lookup settles.
        title = member?.isLoading
          ? t(I18nKey.HOME$LOADING)
          : (member?.email ?? ownerId);
        if (ownerId === currentUserId) {
          title = t(I18nKey.AUTOMATIONS$OWNER_YOU, { owner: title });
        }
      }
      const rank = (() => {
        if (!ownerId) return 3;
        if (group.isServiceAccount) return 2;
        return ownerId === currentUserId ? 0 : 1;
      })();
      return {
        rank,
        section: {
          key: `owner:${ownerId ?? ""}`,
          title,
          automations: group.automations,
        },
      };
    });
    return sections
      .sort(
        (a, b) =>
          a.rank - b.rank || a.section.title.localeCompare(b.section.title),
      )
      .map(({ section }) => section);
  }, [groups, members, serviceAccounts, currentUserId, t]);
}
