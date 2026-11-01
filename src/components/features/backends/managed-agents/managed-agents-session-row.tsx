import { useTranslation } from "react-i18next";
import { Loader2, Pause, Square, Unplug } from "lucide-react";

import {
  isConnectableSession,
  type MarsSession,
} from "#/api/mars/mars-tunnel-backend";
import { BrandButton } from "#/components/features/settings/brand-button";
import type { MarsConnectStage } from "#/hooks/use-connect-mars-session";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { shortSessionId } from "./managed-agents-labels";
import { SessionStatusDot } from "./session-status-dot";
import { useSessionMeta } from "./use-session-meta";

const ICON_BUTTON_CLASS =
  "inline-flex size-8 cursor-pointer items-center justify-center rounded-md text-muted transition-colors hover:bg-interactive-hover hover:text-white disabled:cursor-not-allowed disabled:opacity-40";

interface ManagedAgentsSessionRowProps {
  session: MarsSession;
  isConnected: boolean;
  /** Set while this row's connect flow is running. */
  connectStage: MarsConnectStage | null;
  /** Another row is connecting or this row has a pause or stop in flight. */
  isBusy: boolean;
  error: string | null;
  onConnect: () => void;
  onOpen: () => void;
  onDisconnect: () => void;
  onPause: () => void;
  /** Ends the session for good; the parent confirms before calling. */
  onStop: () => void;
}

export function ManagedAgentsSessionRow({
  session,
  isConnected,
  connectStage,
  isBusy,
  error,
  onConnect,
  onOpen,
  onDisconnect,
  onPause,
  onStop,
}: ManagedAgentsSessionRowProps) {
  const { t } = useTranslation("openhands");
  const { name, phase, meta, stageLabel } = useSessionMeta(
    session,
    connectStage,
  );
  const isConnecting = connectStage !== null;
  const canConnect = isConnectableSession(session);
  // Anything not already over can be stopped, including a failed or stuck
  // provisioning session whose sandbox still needs tearing down.
  const canStop = phase !== "ended";
  const stopButton = canStop ? (
    <button
      type="button"
      onClick={onStop}
      disabled={isBusy}
      aria-label={t(I18nKey.DO_AGENTS$STOP)}
      title={t(I18nKey.DO_AGENTS$STOP)}
      data-testid={`managed-agents-stop-${session.session_id}`}
      className={ICON_BUTTON_CLASS}
    >
      <Square className="size-4" aria-hidden />
    </button>
  ) : null;

  return (
    <li
      data-testid={`managed-agents-session-${session.session_id}`}
      className={cn(
        "flex flex-col gap-1 border-l-2 px-3 py-2.5",
        isConnected ? "border-l-primary" : "border-l-transparent",
      )}
    >
      <div className="flex items-center gap-3">
        <SessionStatusDot phase={phase} />
        <div className="flex min-w-0 flex-1 flex-col">
          <span className="truncate text-sm text-white" title={name}>
            {name}
          </span>
          <span
            className="truncate text-xs text-muted"
            data-testid={`managed-agents-session-status-${session.session_id}`}
          >
            {meta}
            <span className="ml-2 font-mono text-[11px] text-text-tertiary">
              {shortSessionId(session)}
            </span>
          </span>
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          {isConnecting ? (
            <span
              role="status"
              aria-live="polite"
              data-testid={`managed-agents-connecting-${session.session_id}`}
              className="flex items-center gap-2 text-xs text-muted"
            >
              <Loader2 className="size-4 animate-spin" aria-hidden />
              {stageLabel}
            </span>
          ) : null}

          {!isConnecting && isConnected ? (
            <>
              <span className="rounded-full border border-primary/40 px-2 py-0.5 text-[11px] uppercase tracking-wide text-primary">
                {t(I18nKey.DO_AGENTS$CONNECTED)}
              </span>
              <button
                type="button"
                onClick={onDisconnect}
                disabled={isBusy}
                aria-label={t(I18nKey.BUTTON$DISCONNECT)}
                title={t(I18nKey.BUTTON$DISCONNECT)}
                data-testid={`managed-agents-disconnect-${session.session_id}`}
                className={ICON_BUTTON_CLASS}
              >
                <Unplug className="size-4" aria-hidden />
              </button>
              {stopButton}
              <BrandButton
                type="button"
                variant="secondary"
                onClick={onOpen}
                testId={`managed-agents-open-${session.session_id}`}
                className="h-8"
              >
                {t(I18nKey.DO_AGENTS$OPEN)}
              </BrandButton>
            </>
          ) : null}

          {!isConnecting && !isConnected && canConnect ? (
            <>
              {phase === "ready" ? (
                <button
                  type="button"
                  onClick={onPause}
                  disabled={isBusy}
                  aria-label={t(I18nKey.DO_AGENTS$PAUSE)}
                  title={t(I18nKey.DO_AGENTS$PAUSE)}
                  data-testid={`managed-agents-pause-${session.session_id}`}
                  className={ICON_BUTTON_CLASS}
                >
                  <Pause className="size-4" aria-hidden />
                </button>
              ) : null}
              {stopButton}
              <BrandButton
                type="button"
                variant="primary"
                onClick={onConnect}
                isDisabled={isBusy}
                testId={`managed-agents-connect-${session.session_id}`}
                className="h-8"
              >
                {phase === "paused"
                  ? t(I18nKey.DO_AGENTS$RESUME_AND_CONNECT)
                  : t(I18nKey.BACKEND$CONNECT)}
              </BrandButton>
            </>
          ) : null}

          {/* The main action (Open / Connect) is always rightmost, with Stop
              just left of it (rendered above), so the destructive action is
              never mistaken for part of it. */}
          {!isConnecting && !isConnected && !canConnect ? stopButton : null}
        </div>
      </div>

      {error ? (
        <p
          role="alert"
          data-testid={`managed-agents-session-error-${session.session_id}`}
          className="pl-5 text-xs text-status-error whitespace-pre-wrap break-words"
        >
          {error}
        </p>
      ) : null}
    </li>
  );
}
