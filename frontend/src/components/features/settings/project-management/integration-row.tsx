import React from "react";
import { useTranslation } from "react-i18next";

import { useIntegrationStatus } from "#/hooks/query/use-integration-status";
import { useLinkIntegration } from "#/hooks/mutation/use-link-integration";
import { useUnlinkIntegration } from "#/hooks/mutation/use-unlink-integration";
import { useConfigureIntegration } from "#/hooks/mutation/use-configure-integration";
import { I18nKey } from "#/i18n/declaration";
import { ConfigureButton } from "#/components/features/settings/project-management/configure-modal";
import { useModalStore } from "#/stores/modal-store";

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
  const { t } = useTranslation();
  const openModal = useModalStore((state) => state.openModal);
  const closeModal = useModalStore((state) => state.closeModal);

  const { data: integrationData, isLoading: isStatusLoading } =
    useIntegrationStatus(platform);

  const linkMutation = useLinkIntegration(platform, {
    onSettled: () => {
      closeModal();
    },
  });

  const unlinkMutation = useUnlinkIntegration(platform, {
    onSettled: () => {
      closeModal();
    },
  });

  const configureMutation = useConfigureIntegration(platform, {
    onSettled: () => {
      closeModal();
    },
  });

  const handleConfigure = () => {
    openModal("configure-integration", {
      platform,
      platformName,
      integrationData: integrationData || undefined,
      onConfirm: (data: {
        workspace: string;
        webhookSecret: string;
        serviceAccountEmail: string;
        serviceAccountApiKey: string;
        isActive: boolean;
      }) => {
        configureMutation.mutate(data);
      },
      onLink: (workspace: string) => {
        linkMutation.mutate(workspace);
      },
      onUnlink: () => {
        unlinkMutation.mutate();
      },
    });
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
    <div className="flex items-center justify-between" data-testid={dataTestId}>
      <span className="font-medium">{platformName}</span>
      <div className="flex items-center gap-6">
        <ConfigureButton
          onClick={handleConfigure}
          isDisabled={isLoading}
          text={buttonText}
          data-testid={`${platform}-configure-button`}
        />
      </div>
    </div>
  );
}
