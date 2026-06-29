import OpenHandsLogoWhite from "#/assets/branding/openhands-logo-white.svg?react";
import ClaudeMark from "#/assets/branding/claude-mark.svg?react";
import OpenAIMark from "#/assets/branding/openai-mark.svg?react";
import GeminiMark from "#/assets/branding/gemini-mark.svg?react";
import PuzzlePieceIcon from "#/icons/u-puzzle-piece.svg?react";
import type { AgentChipKind } from "#/utils/agent-display-label";

interface AgentChipIconProps {
  kind: AgentChipKind;
  className?: string;
}

const SIZE = 12;

/**
 * Brand mark for the conversation chip. Each harness gets its own recognisable
 * glyph: the OpenHands logo for native conversations, and the relevant provider
 * mark for known ACP servers (Claude, OpenAI/Codex, Gemini). Unknown ACP
 * providers fall back to a generic puzzle piece. The native chip uses the white
 * logo variant, and the OpenAI mark renders in ``currentColor`` (its glyph is
 * monochrome), so both stay legible on the dark chip background; Claude and
 * Gemini keep their signature brand colours.
 */
export function AgentChipIcon({
  kind,
  className = "shrink-0",
}: AgentChipIconProps) {
  switch (kind) {
    case "openhands":
      return (
        <OpenHandsLogoWhite width={SIZE} height={SIZE} className={className} />
      );
    case "acp-claude-code":
      return <ClaudeMark width={SIZE} height={SIZE} className={className} />;
    case "acp-codex":
      return <OpenAIMark width={SIZE} height={SIZE} className={className} />;
    case "acp-gemini-cli":
      return <GeminiMark width={SIZE} height={SIZE} className={className} />;
    default:
      return (
        <PuzzlePieceIcon width={SIZE} height={SIZE} className={className} />
      );
  }
}
