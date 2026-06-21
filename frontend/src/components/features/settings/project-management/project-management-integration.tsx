import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { IntegrationRow } from "./integration-row";
import { JiraDcIntegrationPanel } from "./jira-dc-integration-panel";
import { useConfig } from "#/hooks/query/use-config";

export function ProjectManagementIntegration() {
  const { t } = useTranslation();
  const { data: config } = useConfig();

  const jiraEnabled = config?.feature_flags?.enable_jira;
  const linearEnabled = config?.feature_flags?.enable_linear;
  const jiraDcEnabled = config?.feature_flags?.enable_jira_dc;

  return (
    <div className="flex flex-col gap-6">
      <h3 className="text-xl font-medium text-white">
        {t(I18nKey.PROJECT_MANAGEMENT$TITLE)}
      </h3>

      {/* Jira Cloud + Linear are multi-workspace SaaS integrations and keep the
          compact row + modal. Their config is short. */}
      {(jiraEnabled || linearEnabled) && (
        <div className="flex flex-col gap-4 w-1/4">
          {jiraEnabled && (
            <IntegrationRow
              platform="jira"
              platformName="Jira Cloud"
              data-testid="jira-integration-row"
            />
          )}
          {linearEnabled && (
            <IntegrationRow
              platform="linear"
              platformName="Linear"
              data-testid="linear-integration-row"
            />
          )}
        </div>
      )}

      {/* Jira DC is a single-server integration with more setup, so it gets a
          full-width on-page panel (like the Git webhook managers) instead of a
          cramped modal. */}
      {jiraDcEnabled && <JiraDcIntegrationPanel />}
    </div>
  );
}
