import React from "react";
import { useTranslation } from "react-i18next";
import { useQueryClient } from "@tanstack/react-query";
import {
  getMarsBridge,
  getMarsErrorMessage,
} from "#/api/mars/mars-tunnel-backend";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ErrorAlert } from "#/components/shared/error-alert";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { MARS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";

interface MarsBackendRemovalModalProps {
  name: string;
  sessionId: string;
  /** Drops the backend from the registry (which also closes its tunnel). */
  onRemove: () => void;
  onCancel: () => void;
}

/**
 * Removing a MARS backend only disconnects it: the DigitalOcean session keeps
 * running (and billing) and can be reconnected from Managed Agents. This says
 * so, and offers to stop the session for good instead. As in Managed Agents,
 * the destroy goes first — it can be refused (409/423) while the session is
 * still running, and then the user keeps their connection to it.
 */
export function MarsBackendRemovalModal({
  name,
  sessionId,
  onRemove,
  onCancel,
}: MarsBackendRemovalModalProps) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const [isStopping, setIsStopping] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const canStop = getMarsBridge() !== null;

  const stopSession = async () => {
    const bridge = getMarsBridge();
    if (!bridge) return;
    setIsStopping(true);
    setError(null);
    try {
      await bridge.destroySession(sessionId);
      onRemove();
      void queryClient.invalidateQueries({ queryKey: MARS_QUERY_KEYS.all });
    } catch (stopError) {
      setError(
        getMarsErrorMessage(stopError) ??
          t(I18nKey.DO_AGENTS$SESSION_ACTION_FAILED),
      );
      setIsStopping(false);
    }
  };

  return (
    <ModalBackdrop
      onClose={isStopping ? undefined : onCancel}
      closeOnEscape={!isStopping}
      aria-label={t(I18nKey.BACKEND$REMOVE)}
    >
      <div
        data-testid="mars-backend-removal-modal"
        className="bg-base-secondary p-4 rounded-xl flex flex-col gap-4 border border-border max-w-md"
      >
        <p>{t(I18nKey.BACKEND$REMOVE_MARS_CONFIRMATION, { name })}</p>
        {error ? (
          <ErrorAlert testId="mars-backend-removal-error">{error}</ErrorAlert>
        ) : null}
        <div className="w-full flex flex-wrap justify-end gap-2">
          <BrandButton
            testId="cancel-button"
            type="button"
            variant="secondary"
            onClick={onCancel}
            isDisabled={isStopping}
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
          <BrandButton
            testId="disconnect-only-button"
            type="button"
            variant="secondary"
            onClick={onRemove}
            isDisabled={isStopping}
          >
            {t(I18nKey.BACKEND$DISCONNECT_ONLY)}
          </BrandButton>
          {canStop ? (
            <BrandButton
              testId="stop-session-button"
              type="button"
              variant="danger"
              onClick={stopSession}
              isDisabled={isStopping}
            >
              {t(I18nKey.DO_AGENTS$STOP)}
            </BrandButton>
          ) : null}
        </div>
      </div>
    </ModalBackdrop>
  );
}
