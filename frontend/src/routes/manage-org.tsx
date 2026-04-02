import React from "react";
import { useTranslation } from "react-i18next";
import { useOrganization } from "#/hooks/query/use-organization";
import { useMe } from "#/hooks/query/use-me";
import { useConfig } from "#/hooks/query/use-config";
import { I18nKey } from "#/i18n/declaration";
import { CreditsChip } from "#/ui/credits-chip";
import { InteractiveChip } from "#/ui/interactive-chip";
import { usePermission } from "#/hooks/organizations/use-permissions";
import { createPermissionGuard } from "#/utils/org/permission-guard";
import { isBillingHidden } from "#/utils/org/billing-visibility";
import { ENABLE_ORG_CLAIMS_RESOLVER_ROUTING } from "#/utils/feature-flags";
import { DeleteOrgConfirmationModal } from "#/components/features/org/delete-org-confirmation-modal";
import { GitConversationRouting } from "#/components/features/org/git-conversation-routing";
import { ChangeOrgNameModal } from "#/components/features/org/change-org-name-modal";
import { AddCreditsModal } from "#/components/features/org/add-credits-modal";
import { useBalance } from "#/hooks/query/use-balance";
import { cn } from "#/utils/utils";

export const clientLoader = createPermissionGuard("view_billing");

function ManageOrg() {
  const { t } = useTranslation();
  const { data: me } = useMe();
  const { data: organization } = useOrganization();
  const { data: balance } = useBalance();
  const { data: config } = useConfig();

  const role = me?.role ?? "member";
  const { hasPermission } = usePermission(role);

  const [addCreditsFormVisible, setAddCreditsFormVisible] =
    React.useState(false);
  const [changeOrgNameFormVisible, setChangeOrgNameFormVisible] =
    React.useState(false);
  const [deleteOrgConfirmationVisible, setDeleteOrgConfirmationVisible] =
    React.useState(false);

  const canChangeOrgName = !!me && hasPermission("change_organization_name");
  const canDeleteOrg = !!me && hasPermission("delete_organization");
  const canAddCredits = !!me && hasPermission("add_credits");
  const canManageOrgClaims = !!me && hasPermission("manage_org_claims");
  const shouldHideBilling = isBillingHidden(
    config,
    hasPermission("view_billing"),
  );

  return (
    <div
      data-testid="manage-org-screen"
      className="flex flex-col items-start gap-6"
    >
      {changeOrgNameFormVisible && (
        <ChangeOrgNameModal
          onClose={() => setChangeOrgNameFormVisible(false)}
        />
      )}
      {deleteOrgConfirmationVisible && (
        <DeleteOrgConfirmationModal
          onClose={() => setDeleteOrgConfirmationVisible(false)}
        />
      )}

      {!shouldHideBilling && (
        <div className="flex flex-col gap-2">
          <span className="text-white text-xs font-semibold">
            {t(I18nKey.ORG$CREDITS)}
          </span>
          <div className="flex items-center gap-2">
            <CreditsChip testId="available-credits">
              ${Number(balance ?? 0).toFixed(2)}
            </CreditsChip>
            {canAddCredits && (
              <InteractiveChip onClick={() => setAddCreditsFormVisible(true)}>
                {t(I18nKey.ORG$ADD)}
              </InteractiveChip>
            )}
          </div>
        </div>
      )}

      {addCreditsFormVisible && !shouldHideBilling && (
        <AddCreditsModal onClose={() => setAddCreditsFormVisible(false)} />
      )}

      <div data-testid="org-name" className="flex flex-col gap-2 w-sm">
        <span className="text-white text-xs font-semibold">
          {t(I18nKey.ORG$ORGANIZATION_NAME)}
        </span>

        <div
          className={cn(
            "text-sm p-3 bg-modal-input rounded",
            "flex items-center justify-between",
          )}
        >
          <span className="text-white">{organization?.name}</span>
          {canChangeOrgName && (
            <button
              type="button"
              onClick={() => setChangeOrgNameFormVisible(true)}
              className="text-sm text-org-text font-normal leading-5 hover:text-white transition-colors cursor-pointer"
            >
              {t(I18nKey.ORG$CHANGE)}
            </button>
          )}
        </div>
      </div>

      {canDeleteOrg && (
        <button
          type="button"
          onClick={() => setDeleteOrgConfirmationVisible(true)}
          className="text-xs text-[#FF3B30] cursor-pointer font-semibold hover:underline"
        >
          {t(I18nKey.ORG$DELETE_ORGANIZATION)}
        </button>
      )}

      {canManageOrgClaims && ENABLE_ORG_CLAIMS_RESOLVER_ROUTING() && (
        <GitConversationRouting />
      )}
    </div>
  );
}

export default ManageOrg;
