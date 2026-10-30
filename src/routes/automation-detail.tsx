import { useRef, useState } from "react";
import { useParams, useSearchParams } from "react-router";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import {
  displaySuccessToast,
  displayErrorToast,
} from "#/utils/custom-toast-handlers";
import { getApiErrorMessage } from "#/utils/api-error-message";
import { getErrorStatus } from "#/hooks/query/use-settings";
import { useAutomationDetail } from "#/hooks/query/use-automation-detail";
import {
  useToggleAutomation,
  useDeleteAutomation,
  useDispatchAutomation,
  useChangeRunAs,
} from "#/hooks/query/use-automations";
import {
  useServiceAccountRunAsEnabled,
  useServiceAccounts,
} from "#/hooks/query/use-service-accounts";
import { useAutomationHealth } from "#/hooks/query/use-automation-health";
import { useCloudOrgMember } from "#/hooks/query/use-cloud-org-member";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useNavigation } from "#/context/navigation-context";
import {
  automationListPath,
  hasAutomationInterface,
} from "#/manifests/automation-interface";
import { BackLink } from "#/components/features/automations/detail/back-link";
import { DetailHeader } from "#/components/features/automations/detail/detail-header";
import { AutomationTagsEditor } from "#/components/features/automations/detail/automation-tags-editor";
import { DisabledReasonBanner } from "#/components/features/automations/detail/disabled-reason-banner";
import { PromptSection } from "#/components/features/automations/detail/prompt-section";
import { ScriptSection } from "#/components/features/automations/detail/script-section";
import { ConfigurationSection } from "#/components/features/automations/detail/configuration-section";
import { PluginsSection } from "#/components/features/automations/detail/plugins-section";
import { ActivitySection } from "#/components/features/automations/detail/activity-section";
import { ActivityLogSection } from "#/components/features/automations/detail/activity-log-section";
import { DetailSkeleton } from "#/components/features/automations/detail/detail-skeleton";
import { NotFoundState } from "#/components/features/automations/detail/not-found-state";
import { ErrorState } from "#/components/features/automations/error-state";
import { BackendNotConfigured } from "#/components/features/automations/backend-not-configured";
import { DeleteConfirmationModal } from "#/components/features/automations/delete-confirmation-modal";
import { ChangeRunAsModal } from "#/components/features/automations/change-run-as-modal";
import { useOpenAutomationEditor } from "#/hooks/use-open-automation-editor";
import { useTracking } from "#/hooks/use-tracking";
import { useAutomationAccess } from "#/hooks/use-automation-access";
import {
  useAutomationPermissions,
  useIsAutomationOwner,
} from "#/hooks/use-automation-permissions";
import AutomationService from "#/api/automation-service/automation-service.api";
import type { Automation, ChangeRunAsRequest } from "#/types/automation";
import { isDraftAutomation } from "#/utils/automation-state";
import {
  getAutomationExportFilename,
  serializeAutomation,
} from "#/utils/automation-export";
import { downloadBlob } from "#/utils/utils";

/**
 * Placeholder automation used to keep `useIsAutomationOwner` hooked before
 * the real automation has loaded (rules-of-hooks). Its `user_id` won't match
 * any real user, so the owner check safely returns `false` while loading.
 */
const nullAutomation: Automation = {
  id: "",
  name: "",
  prompt: null,
  trigger: { type: "cron" },
  enabled: false,
  created_at: "",
  updated_at: "",
};

/**
 * The page renders the interface manifest's copy, so without an admitted
 * manifest there is nothing to render: a 404, which the layout's error
 * boundary renders.
 */
export const clientLoader = () => {
  if (!hasAutomationInterface()) {
    throw new Response(null, { status: 404, statusText: "Not Found" });
  }
  return null;
};

export default function AutomationDetail() {
  const { t } = useTranslation("openhands");
  const { automationId } = useParams();
  const [searchParams] = useSearchParams();
  const highlightedRunId = searchParams.get("run");
  const { navigate } = useNavigation();
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showChangeRunAsModal, setShowChangeRunAsModal] = useState(false);
  const { openEditor } = useOpenAutomationEditor();

  const {
    data: healthData,
    isLoading: isHealthLoading,
    refetch: refetchHealth,
  } = useAutomationHealth();

  const isBackendHealthy = healthData?.status === "ok";

  // The automationId in the URL belongs to whichever backend was active
  // when the page first mounted. If the user switches backends, the id
  // is meaningless under the new backend — disable the query so we
  // don't fire a request that the backend selector's redirect will
  // immediately navigate away from anyway.
  const active = useActiveBackend();
  const mountedBackendId = useRef(active.backend.id);
  const backendChanged = mountedBackendId.current !== active.backend.id;

  // Only fetch automation details if the backend is healthy and hasn't changed
  const {
    data: automation,
    isLoading,
    isError,
    error,
    refetch,
  } = useAutomationDetail({
    id: automationId ?? "",
    enabled: isBackendHealthy && !backendChanged,
  });

  const { trackPrebuiltAutomationEnabled, trackAutomationExported } =
    useTracking();
  const toggleMutation = useToggleAutomation();
  const deleteMutation = useDeleteAutomation();
  const dispatchMutation = useDispatchAutomation();
  const changeRunAsMutation = useChangeRunAs();
  // Change Runs As and the service-account name need the service account list,
  // which only managers on a cloud team workspace with the capability can read.
  const serviceAccountRunAsEnabled = useServiceAccountRunAsEnabled();
  const serviceAccountsQuery = useServiceAccounts();
  // Permission hooks must run before any early return (rules-of-hooks). The
  // owner check is a no-op while the automation hasn't loaded yet.
  const { canManage, canEdit, canToggle } = useAutomationAccess(
    automation ?? nullAutomation,
  );
  const isOwner = useIsAutomationOwner(automation ?? nullAutomation);
  const { canManage: hasManagePermission } = useAutomationPermissions();
  // Member lookup for the person on the page: the run-as identity of an
  // automation that runs as a person, or the creator of one that runs as a
  // service account (not an org member, so it has no member lookup).
  // Disabled until the automation has loaded, and on non-cloud backends.
  const runsAsService = automation?.run_as_kind === "service";
  const personId = runsAsService
    ? automation?.created_by_user_id
    : automation?.user_id;
  const personQuery = useCloudOrgMember(personId);

  const is404 = isError && getErrorStatus(error) === 404;

  // Show loading state while checking health
  if (isHealthLoading) {
    return (
      <div className="min-h-full">
        <div className="p-6 max-w-4xl mx-auto">
          <DetailSkeleton />
        </div>
      </div>
    );
  }

  // Show backend not configured state if health check failed
  if (!isBackendHealthy) {
    return (
      <div className="min-h-full">
        <div className="p-6 max-w-4xl mx-auto">
          <BackendNotConfigured onRetry={refetchHealth} />
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="min-h-full">
        <div className="p-6 max-w-4xl mx-auto">
          <DetailSkeleton />
        </div>
      </div>
    );
  }

  if (is404) {
    return (
      <div className="min-h-full">
        <div className="p-6 max-w-4xl mx-auto">
          <NotFoundState />
        </div>
      </div>
    );
  }

  if (isError || !automation) {
    return (
      <div className="min-h-full">
        <div className="p-6 max-w-4xl mx-auto">
          <ErrorState onRetry={() => refetch()} />
        </div>
      </div>
    );
  }

  const handleToggle = () => {
    const willEnable = !automation.enabled;
    toggleMutation.mutate({ id: automation.id, enabled: willEnable });
    if (willEnable) {
      trackPrebuiltAutomationEnabled({
        automationId: automation.id,
        automationName: automation.name,
      });
    }
  };

  const handleDelete = () => {
    deleteMutation.mutate(automation.id, {
      onSuccess: () => {
        navigate?.(automationListPath());
      },
    });
  };

  const handleRunNow = () => {
    dispatchMutation.mutate(automation.id, {
      onSuccess: () => {
        displaySuccessToast(t(I18nKey.AUTOMATIONS$RUN_NOW_SUCCESS));
      },
      onError: (error) => {
        displayErrorToast(
          getApiErrorMessage(error, t(I18nKey.AUTOMATIONS$RUN_NOW_ERROR)),
        );
      },
    });
  };

  const handleChangeRunAs = (body: ChangeRunAsRequest) => {
    changeRunAsMutation.mutate(
      { id: automation.id, body },
      {
        onSuccess: () => {
          setShowChangeRunAsModal(false);
          displaySuccessToast(t(I18nKey.AUTOMATIONS$CHANGE_RUN_AS_SUCCESS));
        },
        onError: (error) => {
          displayErrorToast(
            getApiErrorMessage(
              error,
              t(I18nKey.AUTOMATIONS$CHANGE_RUN_AS_ERROR),
            ),
          );
        },
      },
    );
  };

  const handleExport = () => {
    const contents = `${JSON.stringify(serializeAutomation(automation), null, 2)}\n`;
    downloadBlob(
      new Blob([contents], { type: "application/json" }),
      getAutomationExportFilename(automation),
    );
    trackAutomationExported({ backendKind: active.backend.kind });
  };

  // The service mints run credentials for `automation.user_id`. Cloud only:
  // show the person's email once resolved, fall back to the raw user id when
  // the lookup fails (they left the org, or an app-server without
  // GET /members/{user_id}), nothing while loading.
  let person: string | null = null;
  if (active.backend.kind === "cloud" && personId) {
    person = personQuery.data?.email ?? (personQuery.isError ? personId : null);
  }
  const isDraft = isDraftAutomation(automation);
  // An automation that runs as a service account shows the account's name
  // from the enterprise service account list (managers only). A manager falls
  // back to the raw id when the list is unavailable. A member sees a generic
  // label, never the id.
  const serviceAccountName = runsAsService
    ? (serviceAccountsQuery.data?.find(
        (account) => account.id === automation.user_id,
      )?.name ??
      (hasManagePermission
        ? automation.user_id
        : t(I18nKey.AUTOMATIONS$SERVICE_ACCOUNT_LABEL)))
    : null;
  const runsAs = runsAsService ? serviceAccountName : person;
  const createdBy = runsAsService ? person : null;
  const canChangeRunAs = serviceAccountRunAsEnabled && !isDraft;

  return (
    <div className="min-h-full">
      <div className="p-6 max-w-4xl mx-auto">
        <div className="flex flex-col gap-4">
          <BackLink />
          <DetailHeader
            automation={automation}
            onToggle={handleToggle}
            onEdit={() => openEditor(automation)}
            onDelete={() => setShowDeleteModal(true)}
            onChangeRunAs={
              canChangeRunAs ? () => setShowChangeRunAsModal(true) : undefined
            }
            onExport={handleExport}
            onDownloadTarball={() =>
              AutomationService.downloadTarball(automation.id, automation.name)
            }
            onRunNow={handleRunNow}
            isRunningNow={dispatchMutation.isPending}
            canManage={canManage}
            canEdit={canEdit}
            canToggle={canToggle}
          />
          {/* Tags follow the edit rule: the creator, or a manager of a
              automation that runs as a service account. */}
          <AutomationTagsEditor
            automation={automation}
            canEdit={canEdit && !isDraft}
          />
          <DisabledReasonBanner automation={automation} />
          {automation.prompt ? (
            <PromptSection prompt={automation.prompt} />
          ) : (
            <ScriptSection automation={automation} />
          )}
          <ConfigurationSection
            automation={automation}
            runsAs={runsAs}
            runsAsServiceAccount={runsAsService}
            createdBy={createdBy}
          />
          {automation.plugins && automation.plugins.length > 0 && (
            <PluginsSection plugins={automation.plugins} />
          )}
          <ActivitySection
            createdAt={automation.created_at}
            lastRunAt={automation.last_triggered_at}
          />
          <ActivityLogSection
            automation={automation}
            highlightedRunId={highlightedRunId}
          />
          <ChangeRunAsModal
            automationName={automation.name}
            isOpen={showChangeRunAsModal}
            serviceAccounts={serviceAccountsQuery.data ?? []}
            currentRunAsId={automation.user_id}
            canChangeToSelf={!isOwner}
            isLoading={serviceAccountsQuery.isLoading}
            isError={serviceAccountsQuery.isError}
            isPending={changeRunAsMutation.isPending}
            onConfirm={handleChangeRunAs}
            onCancel={() => setShowChangeRunAsModal(false)}
          />
          <DeleteConfirmationModal
            automationName={automation.name}
            isOpen={showDeleteModal}
            onConfirm={handleDelete}
            onCancel={() => setShowDeleteModal(false)}
          />
        </div>
      </div>
    </div>
  );
}
