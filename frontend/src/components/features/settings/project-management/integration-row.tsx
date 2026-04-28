import React from "react";
import { useTranslation } from "react-i18next";

import { useIntegrationStatus } from "#/hooks/query/use-integration-status";
import { useLinkIntegration } from "#/hooks/mutation/use-link-integration";
import { useUnlinkIntegration } from "#/hooks/mutation/use-unlink-integration";
import { useConfigureIntegration } from "#/hooks/mutation/use-configure-integration";
import { I18nKey } from "#/i18n/declaration";
import {
  ConfigureButton,
  ConfigureModal,
} from "#/components/features/settings/project-management/configure-modal";

interface IntegrationRowProps {
  platform: "jira" | "jira-dc" | "linear";
  platformName: string;
  "data-testid"?: string;
}

export function IntegrationRow({
  platform,
  platformName,
  "data-testid": dataTestId,
}: IntegrationRowProps) {
  const [isConfigureModalOpen, setConfigureModalOpen] = React.useState(false);
  const { t } = useTranslation();

  const { data: integrationData, isLoading: isStatusLoading } =
    useIntegrationStatus(platform);

  const linkMutation = useLinkIntegration(platform, {
    onSettled: () => {
      setConfigureModalOpen(false);
    },
  });

  const unlinkMutation = useUnlinkIntegration(platform, {
    onSettled: () => {
      setConfigureModalOpen(false);
    },
  });

  const configureMutation = useConfigureIntegration(platform, {
    onSettled: () => {
      setConfigureModalOpen(false);
    },
  });

  const handleConfigure = () => {
    setConfigureModalOpen(true);
  };

  const handleLink = (workspace: string) => {
    linkMutation.mutate(workspace);
  };

  const handleUnlink = () => {
    unlinkMutation.mutate();
  };

  const handleConfigureConfirm = (data: {
    workspace: string;
    webhookSecret: string;
    serviceAccountEmail: string;
    serviceAccountApiKey: string;
    isActive: boolean;
  }) => {
    configureMutation.mutate(data);
  };

  const isLoading =
    isStatusLoading ||
    linkMutation.isPending ||
    unlinkMutation.isPending ||
    configureMutation.isPending;

  // Determine if integration is active and workspace exists
  const isIntegrationActive = integrationData?.status === "active";
  const hasWorkspace = integrationData?.workspace;

  // Determine button text based on integration state
  const buttonText =
    isIntegrationActive && hasWorkspace
      ? t(I18nKey.PROJECT_MANAGEMENT$EDIT_BUTTON_LABEL)
      : t(I18nKey.PROJECT_MANAGEMENT$CONFIGURE_BUTTON_LABEL);

  return (
    <div
      className="flex items-center justify-between flex-wrap gap-2"
      data-testid={dataTestId}
    >
      <span className="font-medium">{platformName}</span>
      <div className="flex items-center gap-6">
        <ConfigureButton
          onClick={handleConfigure}
          isDisabled={isLoading}
          text={buttonText}
          data-testid={`${platform}-configure-button`}
        />
      </div>
      <ConfigureModal
        isOpen={isConfigureModalOpen}
        onClose={() => setConfigureModalOpen(false)}
        onConfirm={handleConfigureConfirm}
        onLink={handleLink}
        onUnlink={handleUnlink}
        platformName={platformName}
        platform={platform}
        integrationData={integrationData}
      />
    </div>
  );
}
