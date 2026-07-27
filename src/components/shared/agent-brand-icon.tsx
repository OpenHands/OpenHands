import OpenHandsLogo from "#/assets/branding/openhands-logo.svg?react";
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
 * additionally includes the native OpenHands harness.
 */
export type AgentBrandIconKind = "openhands" | ACPProviderIcon;

// The OpenHands wordmark renders at a 3:2 (width:height) ratio. Kept as a
// named constant so the conversation chip and the onboarding tile (24×16)
// stay visually identical — see ``AgentOptionIcon`` in choose-agent-step.tsx.
const OPENHANDS_LOGO_ASPECT_RATIO = 3 / 2;

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
    // The shipped SVG draws the wordmark with ``fill="white"`` paths but
    // leaves the two hand shapes as ``fill="transparent"`` (negative space).
    // Recolor only the non-transparent paths to ``currentColor`` so the logo
    // inherits the chip's text color *without* filling in the hands — a
    // blanket ``[&_path]`` selector turns the whole mark into a solid blob.
    return (
      <OpenHandsLogo
        width={Math.round(size * OPENHANDS_LOGO_ASPECT_RATIO)}
        height={size}
        className={cn(
          "shrink-0 [&_path:not([fill=transparent])]:fill-current",
          className,
        )}
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
