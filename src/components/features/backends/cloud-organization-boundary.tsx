import React from "react";
import { useTranslation } from "react-i18next";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import { useAllCloudOrganizations } from "#/hooks/query/use-cloud-organizations";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";

export function CloudOrganizationBoundary({
  children,
}: {
  children: React.ReactNode;
}) {
  const { active, setActive } = useActiveBackendContext();
  const organizations = useAllCloudOrganizations();
  const entry = organizations[active.backend.id];
  const isCloud = active.backend.kind === "cloud";
  const hasValidSelection =
    entry?.hasData && entry.orgs.some((org) => org.id === active.orgId);

  // Resolve membership before mounting consumers that send org-scoped requests.
  React.useEffect(() => {
    if (!isCloud || hasValidSelection || !entry?.isSuccess || entry.isFetching)
      return;
    const current = entry.orgs.find((org) => org.id === entry.currentOrgId);
    const personal = entry.orgs.find((org) => org.is_personal === true);
    const target = current ?? personal ?? entry.orgs[0];
    if ((target?.id ?? null) !== active.orgId) {
      setActive(active.backend.id, target?.id ?? null);
    }
  }, [active, entry, hasValidSelection, isCloud, setActive]);

  const canUseSavedSelection = active.orgId && entry?.isError && !entry.hasData;
  if (
    !isCloud ||
    (!entry?.isAuthorizationError &&
      (hasValidSelection || canUseSavedSelection))
  )
    return children;

  if (
    entry &&
    !entry.isFetching &&
    (entry.isError || entry.orgs.length === 0)
  ) {
    return <CloudOrganizationRecovery entry={entry} />;
  }
  return (
    <div className="min-h-full flex items-center justify-center bg-base">
      <LoadingSpinner size="large" />
    </div>
  );
}

function CloudOrganizationRecovery({
  entry,
}: {
  entry: ReturnType<typeof useAllCloudOrganizations>[string];
}) {
  const { t } = useTranslation("openhands");
  const { active, backends, setActive } = useActiveBackendContext();
  return (
    <div className="min-h-full flex flex-col items-center justify-center gap-4 bg-base px-6 text-contrast">
      <p role="alert">
        {t(
          entry.isError
            ? I18nKey.BACKEND$ORGANIZATIONS_LOAD_FAILED
            : I18nKey.BACKEND$ORGANIZATIONS_EMPTY,
        )}
      </p>
      <BrandButton
        type="button"
        variant="primary"
        onClick={() => void entry.refetch()}
      >
        {t(I18nKey.BACKEND$AUTH_RETRY)}
      </BrandButton>
      {backends.length > 1 && (
        <select
          aria-label={t(I18nKey.BACKEND$CHOOSER_TITLE)}
          value={active.backend.id}
          onChange={(event) => setActive(event.target.value, null)}
          className="rounded border border-border bg-base px-3 py-2"
        >
          {backends.map((backend) => (
            <option key={backend.id} value={backend.id}>
              {backend.name}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}
