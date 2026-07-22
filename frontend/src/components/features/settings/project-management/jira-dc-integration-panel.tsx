import React, { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { Typography } from "#/ui/typography";
import { cn } from "#/utils/utils";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { BaseModalTitle } from "#/components/shared/modals/confirmation-modals/base-modal";
import { useConfig } from "#/hooks/query/use-config";
import { useIntegrationStatus } from "#/hooks/query/use-integration-status";
import { useConfigureIntegration } from "#/hooks/mutation/use-configure-integration";
import { useLinkIntegration } from "#/hooks/mutation/use-link-integration";
import { useUnlinkIntegration } from "#/hooks/mutation/use-unlink-integration";
import { useUpdateJiraDcWorkspaceStatus } from "#/hooks/mutation/use-update-jira-dc-workspace-status";
import { useValidateIntegration } from "#/hooks/mutation/use-validate-integration";
import { useMe } from "#/hooks/query/use-me";
import { usePermission } from "#/hooks/organizations/use-permissions";
import { useJiraDcInstanceStatus } from "#/hooks/query/use-jira-dc-instance-status";
import { displaySuccessToast } from "#/utils/custom-toast-handlers";
import { CopyableValue, generateWebhookSecret } from "./configure-modal";

const EMAIL_RE = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

type ModalView = "edit" | "remove" | null;

function buildJiraDcEventsUrl(workspaceId?: number, serverEventsUrl?: string) {
  if (serverEventsUrl) {
    return serverEventsUrl;
  }

  if (!workspaceId) {
    return "";
  }

  const path = `/integration/jira-dc/connections/${workspaceId}/events`;

  return typeof window !== "undefined"
    ? `${window.location.origin}${path}`
    : path;
}

/**
 * On-page Jira Data Center integration. The resting state is a compact table
 * row that mirrors the GitLab / Bitbucket webhook managers above it on the same
 * page (server / service account / status / action). Configuring or editing
 * opens a pop-out modal with a single-column, sequential form. KOTS-managed
 * installs only expose webhook setup in that modal; unmanaged installs also
 * collect server and service-account details. The modal is height-capped + scrollable
 * so it never overflows the viewport. Jira DC is single-server, so there's
 * exactly one connection / service account / webhook to manage.
 */
export function JiraDcIntegrationPanel() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { data: config } = useConfig();
  // OAuth installs already know the host; pre-fill + lock it.
  const jiraDcOAuthHost = config?.jira_dc_oauth_host ?? null;
  const serviceAccountManaged =
    config?.jira_dc_service_account_managed ?? false;
  const managedServiceAccountEmail =
    config?.jira_dc_service_account_email ?? "";
  const serviceAccountConfigError =
    config?.jira_dc_service_account_config_error ?? null;

  const { data: integrationData } = useIntegrationStatus("jira-dc");
  const existingWorkspace = integrationData?.workspace;
  const isWorkspaceEditable = existingWorkspace?.editable ?? false;
  const isActiveIntegration = existingWorkspace
    ? existingWorkspace.status === "active"
    : integrationData?.status === "active";

  // Setting up the connection is admin/owner-only; members just link their own
  // account. (In personal-org mode everyone is an owner, but that path is OAuth
  // -- jira_dc_oauth_host set -- and never routes through the setup form.)
  const { data: me } = useMe();
  const { hasPermission } = usePermission(me?.role ?? "member");
  const canConfigure = hasPermission("manage_integration_providers");

  // A member (email mode, not linked) needs to know whether the connection is
  // set up (and its host) to show "link your account" vs "ask an admin".
  const memberNeedsStatus =
    !existingWorkspace && !jiraDcOAuthHost && !canConfigure;
  const { data: jiraDcInstanceStatus } =
    useJiraDcInstanceStatus(memberNeedsStatus);
  const memberStatusReady =
    memberNeedsStatus && jiraDcInstanceStatus !== undefined;
  const instanceConfigured = jiraDcInstanceStatus?.configured ?? false;
  const instanceHost = jiraDcInstanceStatus?.host ?? null;

  const configureMutation = useConfigureIntegration("jira-dc", {
    onSettled: () => {},
  });
  const linkMutation = useLinkIntegration("jira-dc", {
    onSettled: () => {},
  });
  const unlinkMutation = useUnlinkIntegration("jira-dc", {
    onSettled: () => {},
  });
  const statusMutation = useUpdateJiraDcWorkspaceStatus({
    onSettled: () => {},
  });

  const eventsUrl = buildJiraDcEventsUrl(
    existingWorkspace?.id,
    existingWorkspace?.events_url,
  );

  const [modalView, setModalView] = useState<ModalView>(null);
  const [workspace, setWorkspace] = useState("");
  const [serviceAccountEmail, setServiceAccountEmail] = useState("");
  const [serviceAccountApiKey, setServiceAccountApiKey] = useState("");
  const [adminApiKey, setAdminApiKey] = useState("");
  const [manualMode, setManualMode] = useState(false);
  const [manualSecret, setManualSecret] = useState("");
  const [manualEventsUrl, setManualEventsUrl] = useState("");
  const [manualSetupSaved, setManualSetupSaved] = useState(false);
  const [isActive, setIsActive] = useState(true);
  const [hasSavedApiKey, setHasSavedApiKey] = useState(false);
  const [removeAdminApiKey, setRemoveAdminApiKey] = useState("");
  const [emailError, setEmailError] = useState<string | null>(null);
  const [apiKeyError, setApiKeyError] = useState<string | null>(null);

  // Seed (or re-seed, e.g. when re-opening) form state from the integration.
  const seedForm = React.useCallback(() => {
    if (existingWorkspace) {
      setWorkspace(existingWorkspace.name);
      setServiceAccountEmail(
        serviceAccountManaged
          ? managedServiceAccountEmail
          : (existingWorkspace.svc_acc_email ?? ""),
      );
      setHasSavedApiKey(true);
      setIsActive(existingWorkspace.status === "active");
    } else {
      setWorkspace(jiraDcOAuthHost ?? "");
      setServiceAccountEmail(
        serviceAccountManaged ? managedServiceAccountEmail : "",
      );
      setHasSavedApiKey(serviceAccountManaged);
      setIsActive(true);
    }
    setServiceAccountApiKey("");
    setAdminApiKey("");
    setManualMode(false);
    setEmailError(null);
    setApiKeyError(null);
    setManualEventsUrl("");
    setManualSetupSaved(false);
  }, [
    existingWorkspace,
    jiraDcOAuthHost,
    managedServiceAccountEmail,
    serviceAccountManaged,
  ]);

  React.useEffect(() => {
    seedForm();
  }, [seedForm]);

  const openEdit = () => {
    seedForm();
    if (existingWorkspace && !isActiveIntegration) {
      setIsActive(true);
    }
    setModalView("edit");
  };
  const openRemove = () => {
    seedForm();
    setRemoveAdminApiKey("");
    setModalView("remove");
  };
  const closeModal = () => {
    if (manualSetupSaved) {
      queryClient.invalidateQueries({
        queryKey: ["integration-status", "jira-dc"],
      });
    }
    seedForm();
    setRemoveAdminApiKey("");
    setModalView(null);
  };

  const validateMutation = useValidateIntegration("jira-dc", {
    onSuccess: (data) => {
      if (data.data.status === "active") {
        linkMutation.mutate((jiraDcOAuthHost ?? workspace).trim());
      }
    },
    onError: () => {
      openEdit();
    },
  });
  const isBusy =
    configureMutation.isPending ||
    linkMutation.isPending ||
    unlinkMutation.isPending ||
    validateMutation.isPending ||
    statusMutation.isPending;

  const handleInitialAction = () => {
    // OAuth mode only: the Connect button starts the per-user OAuth link.
    if (jiraDcOAuthHost) {
      validateMutation.mutate(jiraDcOAuthHost);
    }
  };

  const handleEmailChange = (value: string) => {
    if (!existingWorkspace) {
      setManualEventsUrl("");
    }
    setServiceAccountEmail(value);
    setEmailError(
      value && !EMAIL_RE.test(value)
        ? t(I18nKey.PROJECT_MANAGEMENT$SVC_ACC_EMAIL_VALIDATION_ERROR)
        : null,
    );
  };

  const handleApiKeyChange = (value: string) => {
    if (!existingWorkspace) {
      setManualEventsUrl("");
    }
    setServiceAccountApiKey(value);
    setApiKeyError(
      /\s/.test(value)
        ? t(I18nKey.PROJECT_MANAGEMENT$SVC_ACC_API_KEY_VALIDATION_ERROR)
        : null,
    );
  };

  const enableManualMode = () => {
    setManualSecret((prev) => prev || generateWebhookSecret());
    setAdminApiKey("");
    setManualMode(true);
  };

  const handleWorkspaceChange = (value: string) => {
    if (!existingWorkspace) {
      setManualEventsUrl("");
    }
    setWorkspace(value);
  };

  const handleSubmit = () => {
    if (manualMode && !existingWorkspace && manualEventsUrl) {
      statusMutation.mutate(
        {
          workspace,
          isActive: true,
        },
        {
          onSuccess: () => {
            displaySuccessToast(
              t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_WEBHOOK_SETUP_SAVED),
            );
            queryClient.invalidateQueries({
              queryKey: ["integration-status", "jira-dc"],
            });
            seedForm();
            setModalView(null);
          },
        },
      );
      return;
    }

    const shouldGenerateManualDetails = manualMode && !existingWorkspace;

    // Manual mode sends the generated secret the admin is copying into Jira;
    // auto mode sends a blank secret (server-generated) + the one-time admin PAT.
    configureMutation.mutate(
      {
        workspace,
        webhookSecret: manualMode ? manualSecret : "",
        serviceAccountEmail: serviceAccountManaged
          ? managedServiceAccountEmail
          : serviceAccountEmail,
        serviceAccountApiKey: serviceAccountManaged ? "" : serviceAccountApiKey,
        adminApiKey: manualMode ? "" : adminApiKey.trim(),
        isActive: shouldGenerateManualDetails ? false : isActive,
        reloadOnSuccess: false,
        invalidateOnSuccess: !shouldGenerateManualDetails,
      },
      {
        onSuccess: (data) => {
          if (data.redirect) {
            return;
          }

          if (shouldGenerateManualDetails && data.eventsUrl) {
            setManualEventsUrl(data.eventsUrl);
            setManualSetupSaved(true);
            setHasSavedApiKey(true);
            return;
          }

          if (
            !manualMode &&
            adminApiKey.trim() &&
            data.webhookEnrolled === false
          ) {
            return;
          }

          displaySuccessToast(
            t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_WEBHOOK_SETUP_SAVED),
          );
          seedForm();
          setModalView(null);
        },
      },
    );
  };

  const confirmRemove = () => {
    const trimmedAdminApiKey = removeAdminApiKey.trim();
    if (!trimmedAdminApiKey) return;
    unlinkMutation.mutate(trimmedAdminApiKey, {
      onSuccess: () => {
        setRemoveAdminApiKey("");
        setModalView(null);
      },
    });
  };

  const handleActiveToggle = (nextActive: boolean) => {
    setIsActive(nextActive);
    statusMutation.mutate(
      {
        workspace:
          workspace || existingWorkspace?.name || jiraDcOAuthHost || "",
        isActive: nextActive,
      },
      {
        onError: () => setIsActive(!nextActive),
      },
    );
  };

  // PAT required to create a new workspace; optional on edit (blank keeps the
  // stored token). Auto mode on a new workspace needs the one-time admin PAT.
  const apiKeyRequired = !existingWorkspace && !serviceAccountManaged;
  const serviceAccountEmailSatisfied =
    serviceAccountManaged || serviceAccountEmail.trim() !== "";
  const webhookSatisfied = manualMode || adminApiKey.trim() !== "";
  const effectiveManualEventsUrl = manualEventsUrl || eventsUrl;
  const generatedFirstManualSetup =
    manualMode && !existingWorkspace && !!manualEventsUrl;
  let submitButtonLabel = t(I18nKey.PROJECT_MANAGEMENT$CONNECT_BUTTON_LABEL);
  if (manualMode && !existingWorkspace && !manualEventsUrl) {
    submitButtonLabel = t(
      I18nKey.PROJECT_MANAGEMENT$JIRA_DC_GENERATE_WEBHOOK_DETAILS_BUTTON,
    );
  } else if (generatedFirstManualSetup) {
    submitButtonLabel = t(
      I18nKey.PROJECT_MANAGEMENT$JIRA_DC_WEBHOOK_CREATED_BUTTON,
    );
  } else if (existingWorkspace) {
    submitButtonLabel = t(I18nKey.PROJECT_MANAGEMENT$UPDATE_BUTTON_LABEL);
  }
  let manualInstructionKey =
    I18nKey.PROJECT_MANAGEMENT$JIRA_DC_MANUAL_PREPARE_INSTRUCTIONS;
  if (effectiveManualEventsUrl && existingWorkspace) {
    manualInstructionKey =
      I18nKey.PROJECT_MANAGEMENT$JIRA_DC_MANUAL_UPDATE_INSTRUCTIONS;
  } else if (effectiveManualEventsUrl) {
    manualInstructionKey =
      I18nKey.PROJECT_MANAGEMENT$JIRA_DC_MANUAL_INSTRUCTIONS;
  }
  const isSubmitDisabled =
    !workspace.trim() ||
    !serviceAccountEmailSatisfied ||
    (apiKeyRequired && !serviceAccountApiKey.trim()) ||
    emailError !== null ||
    apiKeyError !== null ||
    serviceAccountConfigError !== null ||
    !webhookSatisfied ||
    isBusy;

  const hostLocked = !!existingWorkspace || !!jiraDcOAuthHost;

  const apiKeyPlaceholderKey = hasSavedApiKey
    ? I18nKey.PROJECT_MANAGEMENT$JIRA_DC_SVC_ACC_API_SAVED_PLACEHOLDER
    : I18nKey.PROJECT_MANAGEMENT$JIRA_DC_SVC_ACC_API_PLACEHOLDER;

  const statusBadge = (active = isActiveIntegration) => {
    let label: string;
    let classes: string;
    if (active) {
      label = t(I18nKey.PROJECT_MANAGEMENT$ACTIVE_TOGGLE_LABEL);
      classes = "bg-green-500/20 text-green-400";
    } else {
      label = t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_STATUS_INACTIVE);
      classes = "bg-yellow-500/20 text-yellow-400";
    }
    return (
      <Typography.Text className={cn("px-2 py-1 text-xs rounded", classes)}>
        {label}
      </Typography.Text>
    );
  };

  const sectionLabel = (key: I18nKey) => (
    <span className="text-sm font-medium text-white">{t(key)}</span>
  );

  const colHead =
    "px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider";
  const showServerAndServiceAccountSection = !serviceAccountManaged;

  const serverAndServiceAccountSection = (
    <div className="flex flex-col gap-3">
      <div>
        {sectionLabel(
          I18nKey.PROJECT_MANAGEMENT$JIRA_DC_SERVER_SERVICE_ACCOUNT_SECTION_LABEL,
        )}
        <p className="text-xs text-tertiary-alt mt-1">
          {t(
            serviceAccountManaged
              ? I18nKey.PROJECT_MANAGEMENT$JIRA_DC_SERVICE_ACCOUNT_MANAGED_HELP
              : I18nKey.PROJECT_MANAGEMENT$JIRA_DC_SERVICE_ACCOUNT_SECTION_HELP,
          )}
        </p>
      </div>
      {serviceAccountConfigError && (
        <p className="text-red-500 text-sm">
          {t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_SERVICE_ACCOUNT_CONFIG_ERROR, {
            error: serviceAccountConfigError,
          })}
        </p>
      )}
      <SettingsInput
        testId="jira-dc-host-input"
        label={t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_HOST_LABEL)}
        placeholder={t(
          I18nKey.PROJECT_MANAGEMENT$JIRA_DC_WORKSPACE_NAME_PLACEHOLDER,
        )}
        value={workspace}
        onChange={handleWorkspaceChange}
        className="w-full"
        type="text"
        isDisabled={hostLocked}
      />
      {!hostLocked && (
        <p className="text-xs text-tertiary-alt">
          {t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_HOST_HELP)}
        </p>
      )}
      {serviceAccountManaged ? (
        <SettingsInput
          testId="jira-dc-svc-email-input"
          label={t(I18nKey.PROJECT_MANAGEMENT$SERVICE_ACCOUNT_EMAIL_LABEL)}
          placeholder={t(
            I18nKey.PROJECT_MANAGEMENT$SERVICE_ACCOUNT_EMAIL_PLACEHOLDER,
          )}
          value={managedServiceAccountEmail || "—"}
          className="w-full"
          type="email"
          isDisabled
        />
      ) : (
        <>
          <div>
            <SettingsInput
              testId="jira-dc-svc-email-input"
              label={t(I18nKey.PROJECT_MANAGEMENT$SERVICE_ACCOUNT_EMAIL_LABEL)}
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
              testId="jira-dc-svc-pat-input"
              label={t(
                I18nKey.PROJECT_MANAGEMENT$JIRA_DC_SERVICE_ACCOUNT_API_LABEL,
              )}
              placeholder={t(apiKeyPlaceholderKey)}
              value={serviceAccountApiKey}
              onChange={handleApiKeyChange}
              className="w-full"
              type="password"
              showOptionalTag={hasSavedApiKey}
            />
            {apiKeyError && (
              <p className="text-red-500 text-sm mt-2">{apiKeyError}</p>
            )}
          </div>
        </>
      )}
    </div>
  );

  const webhookSection = (
    <div
      className={cn(
        "flex flex-col gap-3",
        showServerAndServiceAccountSection &&
          "border-t border-neutral-800 pt-4",
      )}
    >
      <div>
        {sectionLabel(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_WEBHOOK_SECTION_LABEL)}
        <p className="text-xs text-tertiary-alt mt-1">
          {t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_WEBHOOK_SECTION_HELP)}
        </p>
      </div>
      <div className="flex w-fit overflow-hidden rounded-sm border border-[#717888] text-sm">
        <button
          type="button"
          data-testid="webhook-mode-auto"
          onClick={() => setManualMode(false)}
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
          onClick={enableManualMode}
          className={`px-3 py-1.5 ${
            manualMode
              ? "bg-[#717888] text-white"
              : "bg-transparent text-tertiary-alt"
          }`}
        >
          {t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_WEBHOOK_MODE_MANUAL)}
        </button>
      </div>
      {!manualMode ? (
        <div>
          <SettingsInput
            testId="admin-api-key-input"
            label={t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_ADMIN_TOKEN_LABEL)}
            placeholder={t(
              I18nKey.PROJECT_MANAGEMENT$JIRA_DC_ADMIN_TOKEN_PLACEHOLDER,
            )}
            value={adminApiKey}
            onChange={setAdminApiKey}
            className="w-full"
            type="password"
            description={
              <p className="text-xs text-tertiary-alt">
                {t(
                  existingWorkspace
                    ? I18nKey.PROJECT_MANAGEMENT$JIRA_DC_EXISTING_ADMIN_TOKEN_HELP
                    : I18nKey.PROJECT_MANAGEMENT$JIRA_DC_ADMIN_TOKEN_HELP,
                )}
              </p>
            }
          />
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          <p className="text-xs text-tertiary-alt">{t(manualInstructionKey)}</p>
          {effectiveManualEventsUrl && (
            <>
              <CopyableValue
                testId="webhook-url-value"
                label={t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_WEBHOOK_URL_LABEL)}
                value={effectiveManualEventsUrl}
              />
              <CopyableValue
                testId="webhook-secret-value"
                label={t(I18nKey.PROJECT_MANAGEMENT$WEBHOOK_SECRET_LABEL)}
                value={manualSecret}
              />
            </>
          )}
        </div>
      )}
    </div>
  );

  // Entry action for a not-yet-linked user. OAuth mode -> Connect your own
  // account. Email mode -> members do nothing (they auto-link on their first
  // @openhands mention, and must never see the admin setup form -- FDE-87);
  // admins set up / manage the connection, with a note on live Jira queries.
  const renderEntryAction = () => {
    if (jiraDcOAuthHost) {
      return (
        <div>
          <BrandButton
            variant="primary"
            onClick={handleInitialAction}
            testId="jira-dc-connect-button"
            type="button"
            isDisabled={isBusy}
          >
            {t(I18nKey.PROJECT_MANAGEMENT$CONNECT_BUTTON_LABEL)}
          </BrandButton>
        </div>
      );
    }
    if (!canConfigure) {
      // Member, email mode. Set up -> a bare "Link account" button (the subtitle
      // carries the prompt). Not set up / still loading -> nothing (the subtitle
      // says "ask an admin"); never the admin setup form (FDE-87).
      if (instanceConfigured) {
        return (
          <div>
            <BrandButton
              variant="primary"
              type="button"
              onClick={() => instanceHost && linkMutation.mutate(instanceHost)}
              testId="jira-dc-link-button"
              isDisabled={isBusy}
            >
              {t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_LINK_BUTTON_LABEL)}
            </BrandButton>
          </div>
        );
      }
      return null;
    }
    return (
      <div>
        <BrandButton
          variant="primary"
          onClick={openEdit}
          testId="jira-dc-configure-button"
          type="button"
          isDisabled={isBusy}
        >
          {t(I18nKey.PROJECT_MANAGEMENT$CONFIGURE_BUTTON_LABEL)}
        </BrandButton>
      </div>
    );
  };

  // Subtitle is state-aware for the not-linked entry: a member sees the link/
  // ask-admin prompt in place of the generic line; an email-mode admin sees the
  // generic line plus the live-query note (folded in, no separate box).
  let subtitleKey = I18nKey.PROJECT_MANAGEMENT$JIRA_DC_PANEL_SUBTITLE;
  if (memberStatusReady) {
    subtitleKey = instanceConfigured
      ? I18nKey.PROJECT_MANAGEMENT$JIRA_DC_MEMBER_LINK_PROMPT
      : I18nKey.PROJECT_MANAGEMENT$JIRA_DC_ASK_ADMIN;
  } else if (canConfigure && !jiraDcOAuthHost) {
    subtitleKey = I18nKey.PROJECT_MANAGEMENT$JIRA_DC_ADMIN_SUBTITLE;
  }

  return (
    <div className="flex flex-col gap-4" data-testid="jira-dc-panel">
      <Typography.H3 className="text-lg font-medium text-white">
        {t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_PLATFORM_NAME)}
      </Typography.H3>
      <Typography.Text className="text-sm text-gray-400">
        {t(subtitleKey)}
      </Typography.Text>

      {existingWorkspace ? (
        <div className="border border-neutral-700 rounded-lg overflow-hidden">
          <table className="w-full">
            <thead className="bg-neutral-800">
              <tr>
                <th className={colHead}>
                  {t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_SERVER_SECTION_LABEL)}
                </th>
                <th className={colHead}>
                  {t(
                    I18nKey.PROJECT_MANAGEMENT$JIRA_DC_SERVICE_ACCOUNT_SECTION_LABEL,
                  )}
                </th>
                <th className={colHead}>
                  {t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_COL_STATUS)}
                </th>
                <th className={colHead}>
                  {t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_COL_ACTION)}
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-700">
              <tr className="hover:bg-neutral-800/50 transition-colors">
                <td className="px-4 py-3">
                  <Typography.Text className="text-sm text-white break-all">
                    {existingWorkspace.name}
                  </Typography.Text>
                  {serviceAccountManaged && (
                    <Typography.Text className="block text-xs text-tertiary-alt mt-1">
                      {t(
                        I18nKey.PROJECT_MANAGEMENT$JIRA_DC_SERVICE_ACCOUNT_MANAGED_BADGE,
                      )}
                    </Typography.Text>
                  )}
                </td>
                <td className="px-4 py-3">
                  <Typography.Text className="text-sm text-gray-300 break-all">
                    {serviceAccountManaged
                      ? managedServiceAccountEmail ||
                        existingWorkspace.svc_acc_email ||
                        "—"
                      : existingWorkspace.svc_acc_email || "—"}
                  </Typography.Text>
                  {serviceAccountManaged && (
                    <Typography.Text className="block text-xs text-tertiary-alt mt-1">
                      {t(
                        I18nKey.PROJECT_MANAGEMENT$JIRA_DC_SERVICE_ACCOUNT_MANAGED_BADGE,
                      )}
                    </Typography.Text>
                  )}
                </td>
                <td className="px-4 py-3">{statusBadge()}</td>
                <td className="px-4 py-3">
                  {isWorkspaceEditable ? (
                    <div className="flex flex-wrap items-center gap-2">
                      {isActiveIntegration ? (
                        <BrandButton
                          variant="danger"
                          onClick={openRemove}
                          testId="remove-integration-button"
                          type="button"
                          isDisabled={isBusy}
                        >
                          {t(
                            I18nKey.PROJECT_MANAGEMENT$JIRA_DC_DISABLE_BUTTON_LABEL,
                          )}
                        </BrandButton>
                      ) : (
                        <BrandButton
                          variant="primary"
                          onClick={openEdit}
                          testId="jira-dc-configure-button"
                          type="button"
                          isDisabled={isBusy}
                        >
                          {t(I18nKey.PROJECT_MANAGEMENT$CONFIGURE_BUTTON_LABEL)}
                        </BrandButton>
                      )}
                    </div>
                  ) : (
                    <div className="flex flex-col items-start gap-2">
                      <BrandButton
                        variant="secondary"
                        onClick={() => unlinkMutation.mutate(undefined)}
                        testId="jira-dc-disconnect-button"
                        type="button"
                        isDisabled={isBusy}
                      >
                        {t(I18nKey.PROJECT_MANAGEMENT$DISCONNECT_BUTTON_LABEL)}
                      </BrandButton>
                      <Typography.Text className="text-xs text-tertiary-alt max-w-56">
                        {t(
                          I18nKey.PROJECT_MANAGEMENT$JIRA_DC_INTEGRATION_OWNER_HELP,
                        )}
                      </Typography.Text>
                    </div>
                  )}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      ) : (
        renderEntryAction()
      )}

      {modalView === "edit" && (
        <ModalBackdrop onClose={closeModal}>
          <ModalBody className="items-start w-[520px] max-h-[85vh] overflow-y-auto gap-4">
            <BaseModalTitle
              title={t(I18nKey.PROJECT_MANAGEMENT$CONFIGURE_MODAL_TITLE, {
                platform: t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_PLATFORM_NAME),
              })}
            />
            <div className="flex flex-col gap-4 w-full">
              {showServerAndServiceAccountSection &&
                serverAndServiceAccountSection}
              {!showServerAndServiceAccountSection &&
                serviceAccountConfigError && (
                  <p className="text-red-500 text-sm">
                    {t(
                      I18nKey.PROJECT_MANAGEMENT$JIRA_DC_SERVICE_ACCOUNT_CONFIG_ERROR,
                      {
                        error: serviceAccountConfigError,
                      },
                    )}
                  </p>
                )}
              {webhookSection}
            </div>

            <div className="flex items-center gap-3 w-full">
              <BrandButton
                variant="primary"
                onClick={handleSubmit}
                testId="jira-dc-submit-button"
                type="button"
                isDisabled={isSubmitDisabled}
              >
                {submitButtonLabel}
              </BrandButton>
              <BrandButton
                variant="secondary"
                onClick={closeModal}
                testId="jira-dc-cancel-button"
                type="button"
                isDisabled={isBusy}
              >
                {t(I18nKey.FEEDBACK$CANCEL_LABEL)}
              </BrandButton>
            </div>
          </ModalBody>
        </ModalBackdrop>
      )}

      {modalView === "remove" && (
        <ModalBackdrop onClose={closeModal}>
          <ModalBody className="items-start w-[460px]">
            <BaseModalTitle
              title={t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_DISABLE_MODAL_TITLE)}
            />
            <div className="flex flex-col gap-4 w-full">
              <div className="flex flex-col gap-3">
                <div className="flex items-center gap-2">
                  {sectionLabel(
                    I18nKey.PROJECT_MANAGEMENT$JIRA_DC_PAUSE_SECTION_LABEL,
                  )}
                  {statusBadge(isActive)}
                </div>
                <div className="flex flex-col gap-1">
                  <SettingsSwitch
                    testId="active-toggle"
                    onToggle={handleActiveToggle}
                    isToggled={isActive}
                    isDisabled={isBusy}
                  >
                    {t(
                      I18nKey.PROJECT_MANAGEMENT$JIRA_DC_EVENT_RESPONSES_TOGGLE_LABEL,
                    )}
                  </SettingsSwitch>
                  <p className="text-xs text-tertiary-alt">
                    {t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_PAUSE_HELP)}
                  </p>
                </div>
              </div>

              <div className="flex flex-col gap-3 border-t border-neutral-800 pt-4">
                {sectionLabel(
                  I18nKey.PROJECT_MANAGEMENT$REMOVE_INTEGRATION_BUTTON_LABEL,
                )}
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
                  description={
                    <p className="text-xs text-tertiary-alt">
                      {t(
                        I18nKey.PROJECT_MANAGEMENT$JIRA_DC_REMOVE_ADMIN_TOKEN_REQUIRED_HELP,
                      )}
                    </p>
                  }
                />
                <div className="flex items-center gap-2 w-full">
                  <BrandButton
                    variant="danger"
                    onClick={confirmRemove}
                    testId="confirm-remove-integration-button"
                    type="button"
                    isDisabled={isBusy || !removeAdminApiKey.trim()}
                  >
                    {t(
                      I18nKey.PROJECT_MANAGEMENT$REMOVE_INTEGRATION_BUTTON_LABEL,
                    )}
                  </BrandButton>
                  <BrandButton
                    variant="secondary"
                    onClick={closeModal}
                    testId="cancel-remove-integration-button"
                    type="button"
                  >
                    {t(I18nKey.SETTINGS_FORM$CLOSE_LABEL)}
                  </BrandButton>
                </div>
              </div>
            </div>
          </ModalBody>
        </ModalBackdrop>
      )}
    </div>
  );
}
