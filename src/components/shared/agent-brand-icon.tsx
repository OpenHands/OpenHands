import RuckusMark from "#/assets/branding/ruckus-mark.svg?react";
import TerminalIcon from "#/icons/terminal.svg?react";
import {
  CLAUDE_CODE_MARK_PATH,
  CLAUDE_CODE_VIEWBOX,
  CODEX_MARK_PATH,
  CODEX_VIEWBOX,
  GEMINI_MARK_PATH,
  GEMINI_VIEWBOX,
} from "#/constants/acp-brand-marks";
import type { ACPProviderIcon } from "#/constants/acp-providers";
import { cn } from "#/utils/utils";

/**
 * Icons the conversation chip + onboarding tiles can render. Strictly broader
 * than {@link ACPProviderIcon} — that type covers ACP CLI subprocesses only
 * (Claude Code, Codex, Gemini, generic terminal fallback), whereas this type
 * additionally includes the native Ruckus harness.
 */
export type AgentBrandIconKind = "openhands" | ACPProviderIcon;

interface AgentBrandIconProps {
  kind: AgentBrandIconKind;
  size?: number;
  className?: string;
  "data-testid"?: string;
}

export function AgentBrandIcon({
  kind,
  size = 12,
  className,
  "data-testid": testId,
}: AgentBrandIconProps) {
  if (kind === "openhands") {
    // Native Ruckus agent: square Ruckus mark.
    return (
      <RuckusMark
        width={size}
        height={size}
        className={cn("shrink-0", className)}
        data-testid={testId ?? "agent-brand-icon-openhands"}
        aria-hidden
      />
    );
  }
  if (kind === "claude-code") {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox={CLAUDE_CODE_VIEWBOX}
        width={size}
        height={size}
        className={cn("shrink-0", className)}
        data-testid={testId ?? "agent-brand-icon-claude-code"}
        aria-hidden
      >
        <path fill="currentColor" d={CLAUDE_CODE_MARK_PATH} />
      </svg>
    );
  }
  if (kind === "codex") {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox={CODEX_VIEWBOX}
        width={size}
        height={size}
        className={cn("shrink-0", className)}
        data-testid={testId ?? "agent-brand-icon-codex"}
        aria-hidden
      >
        <path
          fillRule="evenodd"
          clipRule="evenodd"
          d={CODEX_MARK_PATH}
          fill="currentColor"
        />
      </svg>
    );
  }
  if (kind === "gemini") {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox={GEMINI_VIEWBOX}
        width={size}
        height={size}
        className={cn("shrink-0", className)}
        data-testid={testId ?? "agent-brand-icon-gemini"}
        aria-hidden
      >
        <path fill="currentColor" d={GEMINI_MARK_PATH} />
      </svg>
    );
  }
  return (
    <TerminalIcon
      width={size}
      height={size}
      className={cn("shrink-0", className)}
      data-testid={testId ?? "agent-brand-icon-generic"}
      aria-hidden
    />
  );
}
