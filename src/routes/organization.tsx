import { useTranslation } from "react-i18next";
import { AgentOrganizationChart } from "#/components/features/organization/agent-organization-chart";
import { I18nKey } from "#/i18n/declaration";

export default function OrganizationRoute() {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid="organization-screen"
      className="custom-scrollbar-always h-full overflow-y-auto px-4 py-8 lg:px-10.5"
    >
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <header>
          <h1 className="text-xl font-medium leading-6 text-contrast">
            {t(I18nKey.ORG$TITLE)}
          </h1>
          <p className="mt-1 max-w-3xl text-sm text-text-secondary">
            {t(I18nKey.ORG$SUBTITLE)}
          </p>
        </header>
        <AgentOrganizationChart />
      </div>
    </div>
  );
}
