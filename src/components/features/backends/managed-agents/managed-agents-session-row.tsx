import { useTranslation } from "react-i18next";
import { Loader2, Pause, Unplug } from "lucide-react";

import {
  getSessionPhase,
  type MarsSession,
  type MarsSessionPhase,
} from "#/api/mars/mars-tunnel-backend";
import { BrandButton } from "#/components/features/settings/brand-button";
import type { MarsConnectStage } from "#/hooks/use-connect-mars-session";
import { I18nKey } from "#/i18n/declaration";
import {
  formatRelativeTime,
  isInvalidTimestamp,
} from "#/utils/format-relative-time";
import { cn } from "#/utils/utils";
import { getSessionDisplayName, shortSessionId } from "./managed-agents-labels";

export const PHASE_DOT_CLASS: Record<MarsSessionPhase, string> = {
  ready:
    "bg-[var(--oh-status-success)] shadow-[0_0_0_3px_color-mix(in_srgb,var(--oh-status-success)_22%,transparent)]",
  paused: "bg-[var(--oh-warning)]",
  starting: "bg-[var(--oh-interactive-selected)] animate-pulse",
  failed: "bg-[var(--oh-status-error)]",
  ended: "bg-[var(--oh-text-tertiary)]",
};

export const PHASE_LABEL_KEY: Record<MarsSessionPhase, I18nKey> = {
  ready: I18nKey.DO_AGENTS$STATUS_READY,
  paused: I18nKey.DO_AGENTS$STATUS_PAUSED,
  starting: I18nKey.DO_AGENTS$STATUS_STARTING,
  failed: I18nKey.DO_AGENTS$STATUS_FAILED,
  ended: I18nKey.DO_AGENTS$STATUS_ENDED,
};

export const STAGE_LABEL_KEY: Record<MarsConnectStage, I18nKey> = {
  waking: I18nKey.DO_AGENTS$STAGE_WAKING,
  tunnel: I18nKey.DO_AGENTS$STAGE_TUNNEL,
  agent: I18nKey.DO_AGENTS$STAGE_AGENT,
};

const ICON_BUTTON_CLASS =
  "inline-flex size-8 cursor-pointer items-center justify-center rounded-md text-[var(--oh-muted)] transition-colors hover:bg-[var(--oh-interactive-hover)] hover:text-white disabled:cursor-not-allowed disabled:opacity-40";

interface ManagedAgentsSessionRowProps {
  session: MarsSession;
  isConnected: boolean;
  /** Set while this row's connect flow is running. */
  connectStage: MarsConnectStage | null;
  /** Another row is connecting or this row has a pause in flight. */
  isBusy: boolean;
  error: string | null;
  onConnect: () => void;
  onOpen: () => void;
  onDisconnect: () => void;
  onPause: () => void;
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
}: ManagedAgentsSessionRowProps) {
  const { t, i18n } = useTranslation("openhands");
  const phase = getSessionPhase(session.status);
  const name = getSessionDisplayName(session, t);
  const lastActivity = session.last_event_at ?? session.created_at;
  const meta = [
    t(PHASE_LABEL_KEY[phase]),
    isInvalidTimestamp(lastActivity)
      ? null
      : formatRelativeTime(lastActivity!, i18n.language, t),
  ].filter(Boolean);
  const isConnecting = connectStage !== null;
  const canConnect = phase === "ready" || phase === "paused";

  return (
    <li
      data-testid={`managed-agents-session-${session.session_id}`}
      className={cn(
        "flex flex-col gap-1 border-l-2 px-3 py-2.5",
        isConnected ? "border-l-primary" : "border-l-transparent",
      )}
    >
      <div className="flex items-center gap-3">
        <span
          aria-hidden
          className={cn("size-2 shrink-0 rounded-full", PHASE_DOT_CLASS[phase])}
        />
        <div className="flex min-w-0 flex-1 flex-col">
          <span className="truncate text-sm text-white" title={name}>
            {name}
          </span>
          <span
            className="truncate text-xs text-[var(--oh-muted)]"
            data-testid={`managed-agents-session-status-${session.session_id}`}
          >
            {meta.join(" · ")}
            <span className="ml-2 font-mono text-[11px] text-[var(--oh-text-tertiary)]">
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
              className="flex items-center gap-2 text-xs text-[var(--oh-muted)]"
            >
              <Loader2 className="size-4 animate-spin" aria-hidden />
              {t(STAGE_LABEL_KEY[connectStage])}
            </span>
          ) : null}

          {!isConnecting && isConnected ? (
            <>
              <span className="rounded-full border border-primary/40 px-2 py-0.5 text-[11px] uppercase tracking-wide text-primary">
                {t(I18nKey.DO_AGENTS$CONNECTED)}
              </span>
              <BrandButton
                type="button"
                variant="secondary"
                onClick={onOpen}
                testId={`managed-agents-open-${session.session_id}`}
                className="h-8"
              >
                {t(I18nKey.DO_AGENTS$OPEN)}
              </BrandButton>
              <button
                type="button"
                onClick={onDisconnect}
                disabled={isBusy}
                aria-label={t(I18nKey.DO_AGENTS$DISCONNECT)}
                title={t(I18nKey.DO_AGENTS$DISCONNECT)}
                data-testid={`managed-agents-disconnect-${session.session_id}`}
                className={ICON_BUTTON_CLASS}
              >
                <Unplug className="size-4" aria-hidden />
              </button>
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
                  : t(I18nKey.DO_AGENTS$CONNECT)}
              </BrandButton>
            </>
          ) : null}
        </div>
      </div>

      {error ? (
        <p
          role="alert"
          data-testid={`managed-agents-session-error-${session.session_id}`}
          className="pl-5 text-xs text-[var(--oh-status-error)] whitespace-pre-wrap break-words"
        >
          {error}
        </p>
      ) : null}
    </li>
  );
}
