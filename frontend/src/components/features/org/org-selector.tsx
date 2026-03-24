import React from "react";
import { useTranslation } from "react-i18next";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { useSwitchOrganization } from "#/hooks/mutation/use-switch-organization";
import { useOrganizations } from "#/hooks/query/use-organizations";
import { useShouldHideOrgSelector } from "#/hooks/use-should-hide-org-selector";
import { I18nKey } from "#/i18n/declaration";
import { Organization } from "#/types/org";
import { Dropdown } from "#/ui/dropdown/dropdown";

export function OrgSelector() {
  const { t } = useTranslation();
  const { organizationId } = useSelectedOrganizationId();
  const { data, isLoading } = useOrganizations();
  const organizations = data?.organizations;
  const { mutate: switchOrganization, isPending: isSwitching } =
    useSwitchOrganization();
  const shouldHideSelector = useShouldHideOrgSelector();

  const getOrgDisplayName = React.useCallback(
    (org: Organization) =>
      org.is_personal ? t(I18nKey.ORG$PERSONAL_WORKSPACE) : org.name,
    [t],
  );

  const selectedOrg = React.useMemo(() => {
    if (organizationId) {
      return organizations?.find((org) => org.id === organizationId);
    }

    return organizations?.[0];
  }, [organizationId, organizations]);

  if (shouldHideSelector) {
    return null;
  }

  return (
    <Dropdown
      testId="org-selector"
      key={`${selectedOrg?.id}-${selectedOrg?.name}`}
      defaultValue={{
        label: selectedOrg ? getOrgDisplayName(selectedOrg) : "",
        value: selectedOrg?.id || "",
      }}
      onChange={(item) => {
        if (item && item.value !== organizationId) {
          switchOrganization(item.value);
        }
      }}
      placeholder={t(I18nKey.ORG$SELECT_ORGANIZATION_PLACEHOLDER)}
      loading={isLoading || isSwitching}
      options={
        organizations?.map((org) => ({
          value: org.id,
          label: getOrgDisplayName(org),
        })) || []
      }
      className="bg-[#1F1F1F66] border-[#242424]"
    />
  );
}
