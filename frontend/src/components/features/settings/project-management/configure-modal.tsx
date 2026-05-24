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
import { useConfig } from "#/hooks/query/use-config";

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
      testId={dataTestId}
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

// Generate a URL-safe random secret in the browser for the manual-setup flow,
// so the admin sees the exact value to paste into Jira's webhook config. In
// auto-enroll mode we send an empty secret and the server generates its own.
export function generateWebhookSecret(): string {
  const bytes = new Uint8Array(24);
  crypto.getRandomValues(bytes);
  let binary = "";
  bytes.forEach((b) => {
    binary += String.fromCharCode(b);
  });
  return btoa(binary)
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
}

interface CopyableValueProps {
  label: string;
  value: string;
  testId?: string;
}

// Read-only, selectable value with a copy button - used to surface the webhook
// URL and secret the admin must paste into Jira during manual setup.
export function CopyableValue({ label, value, testId }: CopyableValueProps) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard may be unavailable (insecure context); the value is still
      // selectable by hand.
    }
  };

  return (
    <div className="flex flex-col gap-1">
      <span className="text-sm">{label}</span>
      <div className="flex items-center gap-2">
        <code
          data-testid={testId}
          className="flex-1 select-all break-all bg-tertiary border border-[#717888] rounded-sm p-2 text-xs"
        >
          {value}
        </code>
        <BrandButton
          variant="secondary"
          onClick={handleCopy}
          type="button"
          className="min-w-16"
        >
          {copied
            ? t(I18nKey.PROJECT_MANAGEMENT$COPIED_LABEL)
            : t(I18nKey.PROJECT_MANAGEMENT$COPY_LABEL)}
        </BrandButton>
      </div>
    </div>
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
    adminApiKey: string;
    isActive: boolean;
  }) => void;
  onLink: (workspace: string) => void;
  onUnlink?: (adminApiKey?: string) => void;
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
      // Jira DC only: returned so the form can pre-fill the bot email on edit.
      svc_acc_email?: string;
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
  const { data: config } = useConfig();
  const isJiraDc = platform === "jira-dc";
  // In Jira DC OAuth installs the server host is known from config; pre-fill +
  // lock the host field instead of asking the admin to re-type it.
  const jiraDcOAuthHost = isJiraDc
    ? (config?.jira_dc_oauth_host ?? null)
    : null;
  const [workspace, setWorkspace] = useState("");
  const [webhookSecret, setWebhookSecret] = useState("");
  const [serviceAccountEmail, setServiceAccountEmail] = useState("");
  const [serviceAccountApiKey, setServiceAccountApiKey] = useState("");
  const [isActive, setIsActive] = useState(true);
  const [showConfigurationFields, setShowConfigurationFields] = useState(false);

  // Jira DC only: a one-time admin PAT auto-installs the webhook, and a manual
  // mode reveals the URL + generated secret so an admin can install it by hand.
  const [adminApiKey, setAdminApiKey] = useState("");
  const [manualMode, setManualMode] = useState(false);
  const [manualSecret, setManualSecret] = useState("");
  const [showRemoveConfirm, setShowRemoveConfirm] = useState(false);
  // True when editing a workspace that already has a stored service-account PAT,
  // so the field shows "saved — leave blank to keep" instead of looking empty.
  const [hasSavedApiKey, setHasSavedApiKey] = useState(false);
  // Dedicated, optional admin PAT for the Remove flow (decoupled from the
  // install PAT above): supplying it also revokes the Jira webhook.
  const [removeAdminApiKey, setRemoveAdminApiKey] = useState("");

  const eventsUrl =
    typeof window !== "undefined"
      ? `${window.location.origin}/integration/jira-dc/events`
      : "/integration/jira-dc/events";

  // Determine initial state based on integrationData
  const existingWorkspace = integrationData?.workspace;
  const isWorkspaceEditable = existingWorkspace?.editable ?? false;

  // Validation states
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);
  const [webhookSecretError, setWebhookSecretError] = useState<string | null>(
    null,
  );
  const [emailError, setEmailError] = useState<string | null>(null);
  const [apiKeyError, setApiKeyError] = useState<string | null>(null);

  const resetTransientFields = React.useCallback(() => {
    setWorkspace("");
    setWebhookSecret("");
    setServiceAccountEmail("");
    setServiceAccountApiKey("");
    setAdminApiKey("");
    setManualMode(false);
    setManualSecret("");
    setShowRemoveConfirm(false);
    setHasSavedApiKey(false);
    setRemoveAdminApiKey("");
    setIsActive(false);
    setShowConfigurationFields(false);
    setWorkspaceError(null);
    setWebhookSecretError(null);
    setEmailError(null);
    setApiKeyError(null);
  }, []);

  // Set initial state when the modal opens.
  React.useEffect(() => {
    if (isOpen && existingWorkspace) {
      setWorkspace(existingWorkspace.name);
      setShowConfigurationFields(isWorkspaceEditable);
      // Editing (Jira DC): pre-fill the bot email and mark the PAT as saved
      // (it's never returned), and reflect the stored active state.
      if (isJiraDc && isWorkspaceEditable) {
        setServiceAccountEmail(existingWorkspace.svc_acc_email ?? "");
        setHasSavedApiKey(true);
        setIsActive(existingWorkspace.status === "active");
      }
    } else if (isOpen && !existingWorkspace) {
      // OAuth installs already know the host — pre-fill + lock it below.
      setWorkspace(isJiraDc && jiraDcOAuthHost ? jiraDcOAuthHost : "");
      // Jira DC has no meaningful stage-1 validation gate, so show the full
      // single-stage form immediately. Cloud/Linear keep the two-stage flow.
      setShowConfigurationFields(isJiraDc);
    }
  }, [
    isOpen,
    existingWorkspace,
    isWorkspaceEditable,
    isJiraDc,
    jiraDcOAuthHost,
  ]);

  // Successful configure/remove actions close the modal from the parent. Clear
  // transient secrets here too so one-time admin PATs cannot linger in local UI
  // state while the Settings page remains mounted.
  React.useEffect(() => {
    if (!isOpen) {
      resetTransientFields();
    }
  }, [isOpen, resetTransientFields]);

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

  // Jira DC's "workspace" is the server hostname, so it gets a host-specific
  // label; Cloud/Linear keep the generic "Workspace Name".
  const getWorkspaceLabel = () =>
    isJiraDc
      ? I18nKey.PROJECT_MANAGEMENT$JIRA_DC_HOST_LABEL
      : I18nKey.PROJECT_MANAGEMENT$WORKSPACE_NAME_LABEL;

  // Helper function to get the platform-specific service-account credential label.
  // Jira Cloud issues an "API token", Jira DC a "Personal Access Token (PAT)", and
  // Linear an "API key", so the label must reflect the platform's own terminology.
  const getApiKeyLabel = () => {
    if (platform === "jira") {
      return I18nKey.PROJECT_MANAGEMENT$JIRA_SERVICE_ACCOUNT_API_LABEL;
    }
    if (platform === "jira-dc") {
      return I18nKey.PROJECT_MANAGEMENT$JIRA_DC_SERVICE_ACCOUNT_API_LABEL;
    }
    return I18nKey.PROJECT_MANAGEMENT$SERVICE_ACCOUNT_API_LABEL;
  };

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

  // Reveal the manual-setup view, generating the secret to display once.
  const handleEnableManualMode = () => {
    setManualSecret((prev) => prev || generateWebhookSecret());
    setAdminApiKey("");
    setManualMode(true);
  };

  const handleEnableAutoMode = () => {
    setManualMode(false);
  };

  const confirmAdminRemove = () => {
    // Uses the dedicated Remove-flow PAT, not the install PAT above.
    const trimmedAdminApiKey = removeAdminApiKey.trim();
    onUnlink?.(trimmedAdminApiKey || undefined);
  };

  const handleClose = () => {
    resetTransientFields();
    onClose();
  };

  if (!isOpen) {
    return null;
  }

  const handleConnect = () => {
    if (showConfigurationFields) {
      // For Jira DC the webhook secret is never typed: in manual mode we send
      // the generated secret the admin is copying into Jira; in auto mode we
      // send a blank secret (server-generated) plus the one-time admin PAT.
      let outboundSecret = webhookSecret;
      let outboundAdmin = "";
      if (isJiraDc) {
        if (manualMode) {
          outboundSecret = manualSecret;
        } else {
          outboundSecret = "";
          outboundAdmin = adminApiKey.trim();
        }
      }
      onConfirm({
        workspace,
        webhookSecret: outboundSecret,
        serviceAccountEmail,
        serviceAccountApiKey,
        adminApiKey: outboundAdmin,
        isActive,
      });
    } else if (!existingWorkspace) {
      // First check the workspace with validation for new integrations
      validateMutation.mutate(workspace.trim());
    }
    // For existing workspace that's not editable, no action needed
    // This case shouldn't happen as the button should be hidden
  };

  // For Jira DC the webhook secret is auto-generated, so it is not part of the
  // gate. Auto mode on a brand-new workspace requires the admin PAT; manual
  // mode never requires it; editing an existing workspace requires neither
  // (the admin may just be updating other fields).
  const jiraDcWebhookSatisfied =
    !!existingWorkspace || manualMode || adminApiKey.trim() !== "";

  // The service-account PAT is required to create a new workspace, but optional
  // when editing an existing Jira DC one (blank = keep the stored token).
  const apiKeyRequired = !isJiraDc || !existingWorkspace;
  const baseFieldsInvalid =
    !workspace.trim() ||
    !serviceAccountEmail.trim() ||
    (apiKeyRequired && !serviceAccountApiKey.trim()) ||
    workspaceError !== null ||
    emailError !== null ||
    apiKeyError !== null ||
    validateMutation.isPending;

  // Jira DC uses platform-specific PAT placeholders; when a token is already
  // stored, the field reads "saved — leave blank to keep" rather than empty.
  const apiKeyPlaceholderKey = ((): I18nKey => {
    if (!isJiraDc) {
      return I18nKey.PROJECT_MANAGEMENT$SERVICE_ACCOUNT_API_PLACEHOLDER;
    }
    return hasSavedApiKey
      ? I18nKey.PROJECT_MANAGEMENT$JIRA_DC_SVC_ACC_API_SAVED_PLACEHOLDER
      : I18nKey.PROJECT_MANAGEMENT$JIRA_DC_SVC_ACC_API_PLACEHOLDER;
  })();

  let isConnectDisabled: boolean;
  if (!showConfigurationFields) {
    isConnectDisabled =
      !workspace.trim() ||
      workspaceError !== null ||
      validateMutation.isPending;
  } else if (isJiraDc) {
    isConnectDisabled = baseFieldsInvalid || !jiraDcWebhookSatisfied;
  } else {
    isConnectDisabled =
      baseFieldsInvalid || !webhookSecret.trim() || webhookSecretError !== null;
  }

  const showAdminRemove =
    !!existingWorkspace && isWorkspaceEditable && !!onUnlink;
  const showSelfDisconnect =
    !!existingWorkspace && !isWorkspaceEditable && !!onUnlink;
  const removeWillRevokeWebhook = removeAdminApiKey.trim() !== "";
  const removeHelpKey = I18nKey.PROJECT_MANAGEMENT$JIRA_DC_REMOVE_HELP;
  const removeConfirmKey = removeWillRevokeWebhook
    ? I18nKey.PROJECT_MANAGEMENT$JIRA_DC_REMOVE_WITH_REVOKE_CONFIRM
    : I18nKey.PROJECT_MANAGEMENT$JIRA_DC_REMOVE_WITHOUT_REVOKE_CONFIRM;

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
                    className="text-blue-500 hover:underline"
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
                    className="text-blue-500 underline"
                  >
                    Check the document for more information
                  </a>
                ),
              }}
            />
          )}
          {isJiraDc ? (
            !jiraDcOAuthHost && (
              <p className="mt-4">
                {t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_HOST_HELP)}
              </p>
            )
          ) : (
            <p className="mt-4">
              {t(I18nKey.PROJECT_MANAGEMENT$WORKSPACE_NAME_HINT, {
                platform: platformName,
              })}
            </p>
          )}
        </BaseModalDescription>
        <div className="w-full flex flex-col gap-4 mt-1">
          <div>
            <div className="flex gap-2 items-end">
              <div className="flex-1">
                <SettingsInput
                  label={t(getWorkspaceLabel())}
                  placeholder={t(getWorkspacePlaceholder())}
                  value={workspace}
                  onChange={handleWorkspaceChange}
                  className="w-full"
                  type="text"
                  pattern="^[a-zA-Z0-9\-_.]*$"
                  isDisabled={!!existingWorkspace || !!jiraDcOAuthHost}
                />
              </div>
              {showSelfDisconnect && (
                <BrandButton
                  variant="secondary"
                  onClick={() => onUnlink?.()}
                  testId="unlink-button"
                  type="button"
                  className="mb-0"
                >
                  {t(I18nKey.PROJECT_MANAGEMENT$DISCONNECT_BUTTON_LABEL)}
                </BrandButton>
              )}
            </div>
            {workspaceError && (
              <p className="text-red-500 text-sm mt-2">{workspaceError}</p>
            )}
          </div>

          {showConfigurationFields && (
            <>
              {/* Webhook (Jira -> OpenHands). Jira DC: auto-install via a
                  one-time admin PAT, or reveal the URL + secret for manual
                  setup. Jira Cloud / Linear: a typed webhook secret. */}
              {isJiraDc ? (
                <div className="flex flex-col gap-3">
                  <div>
                    <span className="text-sm font-medium text-white">
                      {t(
                        I18nKey.PROJECT_MANAGEMENT$JIRA_DC_WEBHOOK_SECTION_LABEL,
                      )}
                    </span>
                    <p className="text-xs text-tertiary-alt mt-1">
                      {t(
                        I18nKey.PROJECT_MANAGEMENT$JIRA_DC_WEBHOOK_SECTION_HELP,
                      )}
                    </p>
                  </div>
                  {/* Visible two-option control instead of a buried text link. */}
                  <div className="flex w-fit overflow-hidden rounded-sm border border-[#717888] text-sm">
                    <button
                      type="button"
                      data-testid="webhook-mode-auto"
                      onClick={handleEnableAutoMode}
                      className={`px-3 py-1.5 ${
                        !manualMode
                          ? "bg-[#717888] text-white"
                          : "bg-transparent text-tertiary-alt"
                      }`}
                    >
                      {t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_WEBHOOK_MODE_AUTO)}
                    </button>
                    <button
                      type="button"
                      data-testid="webhook-mode-manual"
                      onClick={handleEnableManualMode}
                      className={`px-3 py-1.5 ${
                        manualMode
                          ? "bg-[#717888] text-white"
                          : "bg-transparent text-tertiary-alt"
                      }`}
                    >
                      {t(
                        I18nKey.PROJECT_MANAGEMENT$JIRA_DC_WEBHOOK_MODE_MANUAL,
                      )}
                    </button>
                  </div>
                  {!manualMode ? (
                    <>
                      <SettingsInput
                        testId="admin-api-key-input"
                        label={t(
                          I18nKey.PROJECT_MANAGEMENT$JIRA_DC_ADMIN_TOKEN_LABEL,
                        )}
                        placeholder={t(
                          I18nKey.PROJECT_MANAGEMENT$JIRA_DC_ADMIN_TOKEN_PLACEHOLDER,
                        )}
                        value={adminApiKey}
                        onChange={setAdminApiKey}
                        className="w-full"
                        type="password"
                        showOptionalTag={!!existingWorkspace}
                      />
                      <p className="text-xs text-tertiary-alt">
                        {t(
                          existingWorkspace
                            ? I18nKey.PROJECT_MANAGEMENT$JIRA_DC_EXISTING_ADMIN_TOKEN_HELP
                            : I18nKey.PROJECT_MANAGEMENT$JIRA_DC_ADMIN_TOKEN_HELP,
                        )}
                      </p>
                    </>
                  ) : (
                    <>
                      <p className="text-xs text-tertiary-alt">
                        {t(
                          existingWorkspace
                            ? I18nKey.PROJECT_MANAGEMENT$JIRA_DC_MANUAL_UPDATE_INSTRUCTIONS
                            : I18nKey.PROJECT_MANAGEMENT$JIRA_DC_MANUAL_INSTRUCTIONS,
                        )}
                      </p>
                      <CopyableValue
                        testId="webhook-url-value"
                        label={t(
                          I18nKey.PROJECT_MANAGEMENT$JIRA_DC_WEBHOOK_URL_LABEL,
                        )}
                        value={eventsUrl}
                      />
                      <CopyableValue
                        testId="webhook-secret-value"
                        label={t(
                          I18nKey.PROJECT_MANAGEMENT$WEBHOOK_SECRET_LABEL,
                        )}
                        value={manualSecret}
                      />
                    </>
                  )}
                </div>
              ) : (
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
                    <p className="text-red-500 text-sm mt-2">
                      {webhookSecretError}
                    </p>
                  )}
                </div>
              )}

              {/* Service account (OpenHands -> Jira): used to post comments and
                  reactions on every event. Required regardless of webhook mode. */}
              {isJiraDc && (
                <div>
                  <span className="text-sm font-medium text-white">
                    {t(
                      I18nKey.PROJECT_MANAGEMENT$JIRA_DC_SERVICE_ACCOUNT_SECTION_LABEL,
                    )}
                  </span>
                  <p className="text-xs text-tertiary-alt mt-1">
                    {t(
                      I18nKey.PROJECT_MANAGEMENT$JIRA_DC_SERVICE_ACCOUNT_SECTION_HELP,
                    )}
                  </p>
                </div>
              )}
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
                  <p className="text-red-500 text-sm mt-2">{emailError}</p>
                )}
              </div>
              <div>
                <SettingsInput
                  label={t(getApiKeyLabel())}
                  placeholder={t(apiKeyPlaceholderKey)}
                  value={serviceAccountApiKey}
                  onChange={handleApiKeyChange}
                  className="w-full"
                  type="password"
                  showOptionalTag={isJiraDc && hasSavedApiKey}
                />
                {apiKeyError && (
                  <p className="text-red-500 text-sm mt-2">{apiKeyError}</p>
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
                {isJiraDc && (
                  <p className="text-xs text-tertiary-alt mt-1">
                    {t(I18nKey.PROJECT_MANAGEMENT$ACTIVE_TOGGLE_HELP)}
                  </p>
                )}
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
              testId="connect-button"
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
          {showAdminRemove && (
            <div className="flex flex-col gap-2">
              <p className="text-xs text-tertiary-alt">
                {t(showRemoveConfirm ? removeConfirmKey : removeHelpKey)}
              </p>
              {showRemoveConfirm ? (
                <>
                  {/* Optional admin PAT scoped to the Remove flow: supplying it
                      also revokes the Jira webhook. Separate from the install
                      PAT in the webhook section so each field has one job. */}
                  {isJiraDc && (
                    <div>
                      <SettingsInput
                        testId="remove-admin-api-key-input"
                        label={t(
                          I18nKey.PROJECT_MANAGEMENT$JIRA_DC_REMOVE_ADMIN_TOKEN_LABEL,
                        )}
                        placeholder={t(
                          I18nKey.PROJECT_MANAGEMENT$JIRA_DC_ADMIN_TOKEN_PLACEHOLDER,
                        )}
                        value={removeAdminApiKey}
                        onChange={setRemoveAdminApiKey}
                        className="w-full"
                        type="password"
                        showOptionalTag
                      />
                      <p className="text-xs text-tertiary-alt mt-1">
                        {t(
                          I18nKey.PROJECT_MANAGEMENT$JIRA_DC_REMOVE_ADMIN_TOKEN_HELP,
                        )}
                      </p>
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-2">
                    <BrandButton
                      variant="danger"
                      onClick={confirmAdminRemove}
                      testId="confirm-remove-integration-button"
                      type="button"
                      className="w-full"
                    >
                      {t(
                        I18nKey.PROJECT_MANAGEMENT$REMOVE_INTEGRATION_BUTTON_LABEL,
                      )}
                    </BrandButton>
                    <BrandButton
                      variant="secondary"
                      onClick={() => {
                        setShowRemoveConfirm(false);
                        setRemoveAdminApiKey("");
                      }}
                      testId="cancel-remove-integration-button"
                      type="button"
                      className="w-full"
                    >
                      {t(I18nKey.FEEDBACK$CANCEL_LABEL)}
                    </BrandButton>
                  </div>
                </>
              ) : (
                <BrandButton
                  variant="danger"
                  onClick={() => setShowRemoveConfirm(true)}
                  testId="remove-integration-button"
                  type="button"
                  className="w-full"
                >
                  {t(
                    I18nKey.PROJECT_MANAGEMENT$REMOVE_INTEGRATION_BUTTON_LABEL,
                  )}
                </BrandButton>
              )}
            </div>
          )}
          <BrandButton
            variant="secondary"
            onClick={handleClose}
            testId="cancel-button"
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
