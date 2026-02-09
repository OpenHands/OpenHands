import React from "react";
import { useTranslation } from "react-i18next";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { useOrganizations } from "#/hooks/query/use-organizations";
import { I18nKey } from "#/i18n/declaration";
import { Organization } from "#/types/org";
import { Dropdown } from "#/ui/dropdown/dropdown";

export function OrgSelector() {
  const { t } = useTranslation();
  const { organizationId, setOrganizationId } = useSelectedOrganizationId();
  const { data: organizations, isLoading } = useOrganizations();

  const getOrgDisplayName = React.useCallback(
    (org: Organization) =>
      org.is_personal ? t(I18nKey.ORG$PERSONAL_WORKSPACE) : org.name,
    [t],
  );

  // Auto-select the first organization when data loads and no org is selected
  React.useEffect(() => {
    if (!organizationId && organizations && organizations.length > 0) {
      setOrganizationId(organizations[0].id);
    }
  }, [organizationId, organizations, setOrganizationId]);

  const selectedOrg = React.useMemo(() => {
    if (organizationId) {
      return organizations?.find((org) => org.id === organizationId);
    }

    return organizations?.[0];
  }, [organizationId, organizations]);

  return (
    <Dropdown
      testId="org-selector"
      key={selectedOrg?.id}
      defaultValue={{
        label: selectedOrg ? getOrgDisplayName(selectedOrg) : "",
        value: selectedOrg?.id || "",
      }}
      onChange={(item) => {
        setOrganizationId(item ? item.value : null);
      }}
      placeholder={t(I18nKey.ORG$SELECT_ORGANIZATION_PLACEHOLDER)}
      loading={isLoading}
      options={
        organizations?.map((org) => ({
          value: org.id,
          label: getOrgDisplayName(org),
        })) || []
      }
    />
  );
}
