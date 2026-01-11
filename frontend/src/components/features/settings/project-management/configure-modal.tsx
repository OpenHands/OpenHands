import React, { useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import {
  BaseModalDescription,
  BaseModalTitle,
} from "#/components/shared/modals/confirmation-modals/base-modal";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { useValidateIntegration } from "#/hooks/mutation/use-validate-integration";

interface ConfigureButtonProps {
  onClick: () => void;
  isDisabled: boolean;
  text?: string;
  "data-testid"?: string;
}

export function ConfigureButton({
  onClick,
  isDisabled,
  text,
  "data-testid": dataTestId,
}: ConfigureButtonProps) {
  const { t } = useTranslation();
  return (
    <BrandButton
      data-testid={dataTestId}
      variant="primary"
      onClick={onClick}
      isDisabled={isDisabled}
      type="button"
      className="w-30 min-w-20"
    >
      {text || t(I18nKey.PROJECT_MANAGEMENT$CONFIGURE_BUTTON_LABEL)}
    </BrandButton>
  );
}

interface ConfigureModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (data: {
    workspace: string;
    webhookSecret: string;
    serviceAccountEmail: string;
    serviceAccountApiKey: string;
    isActive: boolean;
  }) => void;
  onLink: (workspace: string) => void;
  onUnlink?: () => void;
  platformName: string;
  platform: "jira" | "jira-dc" | "linear";
  integrationData?: {
    id: number;
    keycloak_user_id: string;
    status: string;
    workspace?: {
      id: number;
      name: string;
      status: string;
      editable: boolean;
    };
  } | null;
}

export function ConfigureModal({
  isOpen,
  onClose,
  onConfirm,
  onLink,
  onUnlink,
  platformName,
  platform,
  integrationData,
}: ConfigureModalProps) {
  const { t } = useTranslation();
  const [workspace, setWorkspace] = useState("");
  const [webhookSecret, setWebhookSecret] = useState("");
  const [serviceAccountEmail, setServiceAccountEmail] = useState("");
  const [serviceAccountApiKey, setServiceAccountApiKey] = useState("");
  const [isActive, setIsActive] = useState(true);
  const [showConfigurationFields, setShowConfigurationFields] = useState(false);

  // Determine initial state based on integrationData
  const existingWorkspace = integrationData?.workspace;
  const isWorkspaceEditable = existingWorkspace?.editable ?? false;

  // Set initial workspace value when modal opens
  React.useEffect(() => {
    if (isOpen && existingWorkspace) {
      setWorkspace(existingWorkspace.name);
      setShowConfigurationFields(isWorkspaceEditable);
    } else if (isOpen && !existingWorkspace) {
      setWorkspace("");
      setShowConfigurationFields(false);
    }
  }, [isOpen, existingWorkspace, isWorkspaceEditable]);

  // Helper function to get platform-specific placeholder
  const getWorkspacePlaceholder = () => {
    if (platform === "jira") {
      return I18nKey.PROJECT_MANAGEMENT$JIRA_WORKSPACE_NAME_PLACEHOLDER;
    }
    if (platform === "jira-dc") {
      return I18nKey.PROJECT_MANAGEMENT$JIRA_DC_WORKSPACE_NAME_PLACEHOLDER;
    }
    return I18nKey.PROJECT_MANAGEMENT$LINEAR_WORKSPACE_NAME_PLACEHOLDER;
  };

  // Validation states
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);
  const [webhookSecretError, setWebhookSecretError] = useState<string | null>(
    null,
  );
  const [emailError, setEmailError] = useState<string | null>(null);
  const [apiKeyError, setApiKeyError] = useState<string | null>(null);

  const validateMutation = useValidateIntegration(platform, {
    onSuccess: (data) => {
      if (data.data.status === "active") {
        // Validation successful, proceed with linking
        onLink(workspace.trim());
      } else {
        // Show configuration fields for further setup
        setShowConfigurationFields(true);
        setIsActive(true);
      }
    },
    onError: (error) => {
      if (error.response?.status === 404) {
        // Integration not found, show configuration fields
        setShowConfigurationFields(true);
        setIsActive(true);
      } else {
        // Other errors - still show configuration fields as fallback
        setShowConfigurationFields(true);
        setIsActive(true);
      }
    },
  });

  // Validation functions
  const validateWorkspace = (value: string) => {
    const isValid = /^[a-zA-Z0-9\-_.]*$/.test(value);
    if (!isValid && value.length > 0) {
      setWorkspaceError(
        t(I18nKey.PROJECT_MANAGEMENT$WORKSPACE_NAME_VALIDATION_ERROR),
      );
    } else {
      setWorkspaceError(null);
    }
    return isValid;
  };

  const validateWebhookSecret = (value: string) => {
    const hasSpaces = /\s/.test(value);
    if (hasSpaces) {
      setWebhookSecretError(
        t(I18nKey.PROJECT_MANAGEMENT$WEBHOOK_SECRET_NAME_VALIDATION_ERROR),
      );
    } else {
      setWebhookSecretError(null);
    }
    return !hasSpaces;
  };

  const validateEmail = (value: string) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    const isValid = emailRegex.test(value) || value.length === 0;
    if (!isValid && value.length > 0) {
      setEmailError(
        t(I18nKey.PROJECT_MANAGEMENT$SVC_ACC_EMAIL_VALIDATION_ERROR),
      );
    } else {
      setEmailError(null);
    }
    return isValid;
  };

  const validateApiKey = (value: string) => {
    const hasSpaces = /\s/.test(value);
    if (hasSpaces) {
      setApiKeyError(
        t(I18nKey.PROJECT_MANAGEMENT$SVC_ACC_API_KEY_VALIDATION_ERROR),
      );
    } else {
      setApiKeyError(null);
    }
    return !hasSpaces;
  };

  // Input handlers with validation
  const handleWorkspaceChange = (value: string) => {
    setWorkspace(value);
    validateWorkspace(value);
  };

  const handleWebhookSecretChange = (value: string) => {
    setWebhookSecret(value);
    validateWebhookSecret(value);
  };

  const handleEmailChange = (value: string) => {
    setServiceAccountEmail(value);
    validateEmail(value);
  };

  const handleApiKeyChange = (value: string) => {
    setServiceAccountApiKey(value);
    validateApiKey(value);
  };

  const handleClose = () => {
    setWorkspace("");
    setWebhookSecret("");
    setServiceAccountEmail("");
    setServiceAccountApiKey("");
    setIsActive(false);
    setShowConfigurationFields(false);
    setWorkspaceError(null);
    setWebhookSecretError(null);
    setEmailError(null);
    setApiKeyError(null);
    onClose();
  };

  if (!isOpen) {
    return null;
  }

  const handleConnect = () => {
    if (showConfigurationFields) {
      // Full configuration flow (either new configuration or editing existing)
      onConfirm({
        workspace,
        webhookSecret,
        serviceAccountEmail,
        serviceAccountApiKey,
        isActive,
      });
    } else if (!existingWorkspace) {
      // First check the workspace with validation for new integrations
      validateMutation.mutate(workspace.trim());
    }
    // For existing workspace that's not editable, no action needed
    // This case shouldn't happen as the button should be hidden
  };

  const isConnectDisabled = showConfigurationFields
    ? !workspace.trim() ||
      !webhookSecret.trim() ||
      !serviceAccountEmail.trim() ||
      !serviceAccountApiKey.trim() ||
      workspaceError !== null ||
      webhookSecretError !== null ||
      emailError !== null ||
      apiKeyError !== null ||
      validateMutation.isPending
    : !workspace.trim() ||
      workspaceError !== null ||
      validateMutation.isPending;

  return (
    <ModalBackdrop onClose={handleClose}>
      <ModalBody className="items-start border border-tertiary w-96">
        <BaseModalTitle
          title={
            showConfigurationFields
              ? t(I18nKey.PROJECT_MANAGEMENT$CONFIGURE_MODAL_TITLE, {
                  platform: platformName,
                })
              : t(I18nKey.PROJECT_MANAGEMENT$LINK_CONFIRMATION_TITLE)
          }
        />
        <BaseModalDescription>
          {showConfigurationFields ? (
            <Trans
              i18nKey={
                I18nKey.PROJECT_MANAGEMENT$CONFIGURE_MODAL_DESCRIPTION_STAGE_2
              }
              components={{
                b: <b />,
                a: (
                  <a
                    href="https://docs.all-hands.dev/usage/cloud/openhands-cloud"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-accent hover:underline"
                  >
                    Check the document for more information
                  </a>
                ),
              }}
            />
          ) : (
            <Trans
              i18nKey={
                I18nKey.PROJECT_MANAGEMENT$CONFIGURE_MODAL_DESCRIPTION_STAGE_1
              }
              components={{
                b: <b />,
                a: (
                  <a
                    href="https://docs.all-hands.dev/usage/cloud/openhands-cloud"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-accent underline"
                  >
                    Check the document for more information
                  </a>
                ),
              }}
            />
          )}
          <p className="mt-4">
            {t(I18nKey.PROJECT_MANAGEMENT$WORKSPACE_NAME_HINT, {
              platform: platformName,
            })}
          </p>
        </BaseModalDescription>
        <div className="w-full flex flex-col gap-4 mt-1">
          <div>
            <div className="flex gap-2 items-end">
              <div className="flex-1">
                <SettingsInput
                  label={t(I18nKey.PROJECT_MANAGEMENT$WORKSPACE_NAME_LABEL)}
                  placeholder={t(getWorkspacePlaceholder())}
                  value={workspace}
                  onChange={handleWorkspaceChange}
                  className="w-full"
                  type="text"
                  pattern="^[a-zA-Z0-9\-_.]*$"
                  isDisabled={!!existingWorkspace}
                />
              </div>
              {existingWorkspace && onUnlink && (
                <BrandButton
                  variant="secondary"
                  onClick={onUnlink}
                  data-testid="unlink-button"
                  type="button"
                  className="mb-0"
                >
                  {t(I18nKey.PROJECT_MANAGEMENT$UNLINK_BUTTON_LABEL)}
                </BrandButton>
              )}
            </div>
            {workspaceError && (
              <p className="text-danger text-sm mt-2">{workspaceError}</p>
            )}
          </div>

          {showConfigurationFields && (
            <>
              <div>
                <SettingsInput
                  label={t(I18nKey.PROJECT_MANAGEMENT$WEBHOOK_SECRET_LABEL)}
                  placeholder={t(
                    I18nKey.PROJECT_MANAGEMENT$WEBHOOK_SECRET_PLACEHOLDER,
                  )}
                  value={webhookSecret}
                  onChange={handleWebhookSecretChange}
                  className="w-full"
                  type="password"
                />
                {webhookSecretError && (
                  <p className="text-danger text-sm mt-2">
                    {webhookSecretError}
                  </p>
                )}
              </div>
              <div>
                <SettingsInput
                  label={t(
                    I18nKey.PROJECT_MANAGEMENT$SERVICE_ACCOUNT_EMAIL_LABEL,
                  )}
                  placeholder={t(
                    I18nKey.PROJECT_MANAGEMENT$SERVICE_ACCOUNT_EMAIL_PLACEHOLDER,
                  )}
                  value={serviceAccountEmail}
                  onChange={handleEmailChange}
                  className="w-full"
                  type="email"
                />
                {emailError && (
                  <p className="text-danger text-sm mt-2">{emailError}</p>
                )}
              </div>
              <div>
                <SettingsInput
                  label={t(
                    I18nKey.PROJECT_MANAGEMENT$SERVICE_ACCOUNT_API_LABEL,
                  )}
                  placeholder={t(
                    I18nKey.PROJECT_MANAGEMENT$SERVICE_ACCOUNT_API_PLACEHOLDER,
                  )}
                  value={serviceAccountApiKey}
                  onChange={handleApiKeyChange}
                  className="w-full"
                  type="password"
                />
                {apiKeyError && (
                  <p className="text-danger text-sm mt-2">{apiKeyError}</p>
                )}
              </div>
              <div className="mt-4">
                <SettingsSwitch
                  testId="active-toggle"
                  onToggle={setIsActive}
                  isToggled={isActive}
                >
                  {t(I18nKey.PROJECT_MANAGEMENT$ACTIVE_TOGGLE_LABEL)}
                </SettingsSwitch>
              </div>
            </>
          )}
        </div>
        <div className="flex flex-col gap-2 w-full mt-4">
          {/* Hide the connect/edit button if workspace exists but is not editable */}
          {(!existingWorkspace || isWorkspaceEditable) && (
            <BrandButton
              variant="primary"
              onClick={handleConnect}
              data-testid="connect-button"
              type="button"
              className="w-full"
              isDisabled={isConnectDisabled}
            >
              {(() => {
                if (existingWorkspace && showConfigurationFields) {
                  return t(I18nKey.PROJECT_MANAGEMENT$UPDATE_BUTTON_LABEL);
                }
                return t(I18nKey.PROJECT_MANAGEMENT$CONNECT_BUTTON_LABEL);
              })()}
            </BrandButton>
          )}
          <BrandButton
            variant="secondary"
            onClick={handleClose}
            data-testid="cancel-button"
            type="button"
            className="w-full"
          >
            {t(I18nKey.FEEDBACK$CANCEL_LABEL)}
          </BrandButton>
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
