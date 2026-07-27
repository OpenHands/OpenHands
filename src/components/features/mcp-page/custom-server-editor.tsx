import React from "react";
import { AxiosError } from "axios";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import {
  MCPServerForm,
  type TestMessage,
} from "#/components/features/settings/mcp-settings/mcp-server-form";
import { seedMcpServerHealth } from "#/api/mcp-health/probe-mcp-server-health";
import { useAddMcpServer } from "#/hooks/mutation/use-add-mcp-server";
import { useUpdateMcpServer } from "#/hooks/mutation/use-update-mcp-server";
import { useDeleteMcpServer } from "#/hooks/mutation/use-delete-mcp-server";
import { useTestMcpServer } from "#/hooks/mutation/use-test-mcp-server";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { ExtendedMCPTestResponse, MCPServerConfig } from "#/types/mcp-server";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { makeMcpTestErrorMessage } from "#/utils/mcp-test-error-message";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { cn } from "#/utils/utils";
import { modalTitleLgClassName } from "#/utils/modal-classes";
import McpService from "#/api/mcp-service/mcp-service.api";

interface CustomServerEditorProps {
  server: MCPServerConfig;
  existingServers: MCPServerConfig[];
  onClose: () => void;
}

/**
 * Modal wrapper around `MCPServerForm` so users can hand-author
 * arbitrary stdio / SSE / SHTTP entries without reaching for raw JSON.
 * An empty `server.id` means "Add new".
 */
export function CustomServerEditor({
  server,
  existingServers,
  onClose,
}: CustomServerEditorProps) {
  const { t } = useTranslation("openhands");
  const { mutate: addMcpServer, isPending: isAdding } = useAddMcpServer();
  const { mutate: updateMcpServer, isPending: isUpdating } =
    useUpdateMcpServer();
  const { mutate: deleteMcpServer, isPending: isDeleting } =
    useDeleteMcpServer();
  const {
    mutate: testServer,
    isPending: isTesting,
    data: testResult,
    reset: resetTest,
  } = useTestMcpServer();
  const [showDeleteConfirm, setShowDeleteConfirm] = React.useState(false);
  const [oauthTestResult, setOauthTestResult] =
    React.useState<ExtendedMCPTestResponse | null>(null);
  const [isOauthTesting, setIsOauthTesting] = React.useState(false);

  // The MCP connectivity-test endpoint only exists on the local agent-server.
  // For cloud backends `McpService.testServer` short-circuits with a synthetic
  // success so the save still completes; we hide the manual "Test connection"
  // button here so cloud users aren't shown a misleading "0 tools" result.
  const { backend } = useActiveBackend();
  const isCloudBackend = backend.kind === "cloud";

  const isEditing = !!server.id;
  const isPending = isAdding || isUpdating || isDeleting;
  const isDismissBlocked =
    isPending || isTesting || isOauthTesting || showDeleteConfirm;

  const testMessage: TestMessage | null = React.useMemo(() => {
    const result = oauthTestResult ?? testResult;
    if (!result) return null;
    if (result.ok) {
      return {
        ok: true,
        text: t(I18nKey.MCP$TEST_SUCCESS, { count: result.tools.length }),
      };
    }
    return {
      ok: false,
      text: makeMcpTestErrorMessage(t, result.error_kind, result.error),
    };
  }, [oauthTestResult, testResult, t]);

  // A save always follows a fresh successful probe of the exact config being
  // saved, so publish that result to the card's health entry. On cloud
  // backends the probe is synthetic — never present it as a health verdict.
  const seedSavedServerHealth = (
    serverToSave: MCPServerConfig,
    result: ExtendedMCPTestResponse,
  ) => {
    if (isCloudBackend) return;
    // When editing, the server's own entry must be overwritten, so only the
    // OTHER servers guard against a same-key collision.
    const otherServers = isEditing
      ? existingServers.filter((existing) => existing.id !== server.id)
      : existingServers;
    seedMcpServerHealth(serverToSave, result, otherServers);
  };

  // Shared error handler so both add and update surface backend errors
  // as a toast instead of failing silently — previously these calls
  // had no `onError` and the modal closed even on a 4xx/5xx, leaving
  // the user to discover the failure on the next page load.
  const handleError = (err: unknown) => {
    const message = retrieveAxiosErrorMessage(err as AxiosError);
    displayErrorToast(message || t(I18nKey.ERROR$GENERIC));
  };

  const handleSubmit = (payload: MCPServerConfig) => {
    resetTest();
    setOauthTestResult(null);
    if (payload.auth?.strategy === "oauth2") {
      setIsOauthTesting(true);
      void McpService.authorizeOAuth(payload)
        .then((result) => {
          setOauthTestResult(result);
          if (!result.ok) return;
          const serverToSave = result.oauth_state
            ? {
                ...payload,
                auth: { ...payload.auth!, state: result.oauth_state },
              }
            : payload;
          const onSaved = () => {
            seedSavedServerHealth(serverToSave, result);
            onClose();
          };
          if (isEditing) {
            updateMcpServer(
              { serverId: server.id, server: serverToSave },
              { onSuccess: onSaved, onError: handleError },
            );
          } else {
            addMcpServer(serverToSave, {
              onSuccess: onSaved,
              onError: handleError,
            });
          }
        })
        .catch(handleError)
        .finally(() => setIsOauthTesting(false));
      return;
    }
    testServer(payload, {
      onSuccess: (result) => {
        if (!result.ok) {
          // Test failed — modal stays open, error shown via testMessage.
          return;
        }
        const serverToSave =
          result.oauth_state && payload.auth?.strategy === "oauth2"
            ? {
                ...payload,
                auth: { ...payload.auth, state: result.oauth_state },
              }
            : payload;
        const onSaved = () => {
          seedSavedServerHealth(serverToSave, result);
          onClose();
        };
        if (isEditing) {
          updateMcpServer(
            { serverId: server.id, server: serverToSave },
            { onSuccess: onSaved, onError: handleError },
          );
        } else {
          addMcpServer(serverToSave, {
            onSuccess: onSaved,
            onError: handleError,
          });
        }
      },
      onError: handleError,
    });
  };

  const handleTestClick = (payload: MCPServerConfig) => {
    setOauthTestResult(null);
    if (payload.auth?.strategy === "oauth2" && !isCloudBackend) {
      setIsOauthTesting(true);
      void McpService.authorizeOAuth(payload)
        .then(setOauthTestResult)
        .catch(handleError)
        .finally(() => setIsOauthTesting(false));
      return;
    }
    testServer(payload);
  };

  const handleConfirmDelete = () => {
    deleteMcpServer(server, {
      onSuccess: () => {
        displaySuccessToast(t(I18nKey.MCP$REMOVE_SUCCESS));
        setShowDeleteConfirm(false);
        onClose();
      },
      onError: (err) => {
        handleError(err);
        setShowDeleteConfirm(false);
      },
    });
  };

  return (
    <>
      <ModalBackdrop
        // Block backdrop-click / Escape from dismissing the modal while
        // a mutation is in flight — closing mid-request would orphan
        // the request and leave the user with no error feedback.
        onClose={isDismissBlocked ? undefined : onClose}
        closeOnEscape={!isDismissBlocked}
        aria-label={
          isEditing
            ? t(I18nKey.MCP$EDIT_CUSTOM_TITLE)
            : t(I18nKey.MCP$ADD_CUSTOM_TITLE)
        }
      >
        <div
          data-testid="mcp-custom-editor"
          className="relative bg-base-secondary p-6 rounded-xl border border-[var(--oh-border)] w-[520px] max-w-[90vw] max-h-[90vh] overflow-y-auto custom-scrollbar"
        >
          <ModalCloseButton
            onClose={onClose}
            testId="mcp-custom-editor-close"
            disabled={isDismissBlocked}
          />
          <h2 className={cn("mb-4 pr-6", modalTitleLgClassName)}>
            {isEditing
              ? t(I18nKey.MCP$EDIT_CUSTOM_TITLE)
              : t(I18nKey.MCP$ADD_CUSTOM_TITLE)}
          </h2>
          <MCPServerForm
            mode={isEditing ? "edit" : "add"}
            server={isEditing ? server : undefined}
            existingServers={existingServers}
            onSubmit={handleSubmit}
            onCancel={onClose}
            onDelete={isEditing ? () => setShowDeleteConfirm(true) : undefined}
            isActionDisabled={isPending}
            onTest={isCloudBackend ? undefined : handleTestClick}
            isTestPending={isTesting || isOauthTesting}
            testMessage={isCloudBackend ? null : testMessage}
          />
        </div>
      </ModalBackdrop>

      {showDeleteConfirm ? (
        <ConfirmationModal
          text={t(I18nKey.SETTINGS$MCP_CONFIRM_DELETE)}
          onCancel={() => setShowDeleteConfirm(false)}
          onConfirm={handleConfirmDelete}
          isConfirming={isDeleting}
        />
      ) : null}
    </>
  );
}
