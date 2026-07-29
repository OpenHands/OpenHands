import { useTranslation } from "react-i18next";
import { FaArchive } from "react-icons/fa";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { SandboxStatus } from "#/api/conversation-service/agent-server-conversation-service.types";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";

interface ConversationStatusDotProps {
  executionStatus: ExecutionStatus | null | undefined;
  /**
   * Cloud-only sandbox lifecycle status. When provided, MISSING and ERROR
   * override the execution-status visual so the dot reflects the sandbox
   * state rather than the last agent execution state.
   */
  sandboxStatus?: SandboxStatus | null;
  /**
   * Wrap the dot in a tooltip showing the human-readable status label.
   * Disable this when the dot is already nested inside a larger tooltip
   * (e.g. the collapsed-sidebar conversation preview) so the smaller
   * tooltip doesn't intercept the hover.
   */
  showTooltip?: boolean;
}

type Visual = "check" | "working" | "active" | "paused" | "error" | "unknown";

const visualFor = (status: ExecutionStatus | null | undefined): Visual => {
  switch (status) {
    case ExecutionStatus.FINISHED:
      return "check";
    case ExecutionStatus.RUNNING:
      return "working";
    case ExecutionStatus.IDLE:
    case ExecutionStatus.WAITING_FOR_CONFIRMATION:
      return "active";
    case ExecutionStatus.PAUSED:
      return "paused";
    case ExecutionStatus.ERROR:
    case ExecutionStatus.STUCK:
      return "error";
    default:
      return "unknown";
  }
};

const labelKeyFor = (visual: Visual, isArchived?: boolean): string => {
  if (isArchived) return "COMMON$ARCHIVED";
  switch (visual) {
    case "check":
      return "COMMON$FINISHED";
    case "working":
    case "active":
      return "COMMON$WORKING";
    case "paused":
      return "COMMON$PAUSED";
    case "error":
      return "COMMON$ERROR";
    default:
      return "COMMON$STOPPED";
  }
};

function renderIndicator(visual: Visual) {
  switch (visual) {
    case "check":
      return (
        <svg
          data-testid="conversation-status-check"
          viewBox="0 0 12 12"
          className="w-2.5 h-2.5 stroke-[var(--oh-status-success)]"
          fill="none"
          strokeWidth={2.25}
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M2.5 6.5 5 9l4.5-5.5" />
        </svg>
      );
    case "working":
      return (
        <span
          data-testid="conversation-status-working"
          className="w-1.5 h-1.5 rounded-full animate-pulse bg-[var(--oh-status-success)]"
        />
      );
    case "active":
      return (
        <span
          data-testid="conversation-status-active"
          className="w-1.5 h-1.5 rounded-full bg-[var(--oh-status-success)]"
        />
      );
    case "paused":
      return (
        <span
          data-testid="conversation-status-paused"
          className="w-1.5 h-1.5 rounded-full bg-[var(--oh-muted)]"
        />
      );
    case "error":
      return (
        <span
          data-testid="conversation-status-error"
          className="w-1.5 h-1.5 rounded-full bg-[var(--oh-status-error)]"
        />
      );
    default:
      return (
        <span
          data-testid="conversation-status-unknown"
          className="w-1.5 h-1.5 rounded-full bg-[var(--oh-color-tertiary)]"
        />
      );
  }
}

export function ConversationStatusDot({
  executionStatus,
  sandboxStatus,
  showTooltip = true,
}: ConversationStatusDotProps) {
  const { t } = useTranslation("openhands");

  // sandbox_status === "MISSING" → show archived (gray) dot
  // sandbox_status === "ERROR"   → show error (red) dot
  // Otherwise fall through to the execution-status visual.
  const isArchived = sandboxStatus === "MISSING";
  const effectiveVisual: Visual =
    sandboxStatus === "ERROR"
      ? "error"
      : isArchived
        ? "paused"
        : visualFor(executionStatus);

  const visual = effectiveVisual;
  const label = t(labelKeyFor(visual, isArchived));
  const indicator = isArchived ? (
    <FaArchive
      data-testid="conversation-status-archived"
      size={10}
      className="shrink-0 text-[var(--oh-muted)] opacity-60"
      aria-hidden
    />
  ) : (
    renderIndicator(visual)
  );

  const dot = (
    <div className="w-2.5 h-2.5 flex items-center justify-center shrink-0">
      {indicator}
    </div>
  );

  if (!showTooltip) return dot;

  return (
    <StyledTooltip
      content={label}
      placement="right"
      showArrow
      tooltipClassName="bg-base text-white text-xs shadow-lg"
    >
      {dot}
    </StyledTooltip>
  );
}
