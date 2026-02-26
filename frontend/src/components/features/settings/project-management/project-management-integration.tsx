import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { IntegrationRow } from "./integration-row";
import { useConfig } from "#/hooks/query/use-config";

export function ProjectManagementIntegration() {
  const { t } = useTranslation();
  const { data: config } = useConfig();

  return (
    <div className="flex flex-col gap-4 w-1/4">
      <h3 className="text-xl font-medium text-white">
        {t(I18nKey.PROJECT_MANAGEMENT$TITLE)}
      </h3>
      <div className="flex flex-col gap-4">
        {config?.feature_flags?.enable_jira && (
          <IntegrationRow
            platform="jira"
            platformName="Jira Cloud"
            data-testid="jira-integration-row"
          />
        )}
        {config?.feature_flags?.enable_jira_dc && (
          <IntegrationRow
            platform="jira-dc"
            platformName="Jira Data Center"
            data-testid="jira-dc-integration-row"
          />
        )}
        {config?.feature_flags?.enable_linear && (
          <IntegrationRow
            platform="linear"
            platformName="Linear"
            data-testid="linear-integration-row"
          />
        )}
      </div>
    </div>
  );
}
