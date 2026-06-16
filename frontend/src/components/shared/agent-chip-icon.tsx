import OpenHandsLogo from "#/assets/branding/openhands-logo.svg?react";
import PuzzlePieceIcon from "#/icons/u-puzzle-piece.svg?react";
import type { AgentChipKind } from "#/utils/agent-display-label";

interface AgentChipIconProps {
  kind: AgentChipKind;
  className?: string;
}

const SIZE = 12;

export function AgentChipIcon({
  kind,
  className = "shrink-0",
}: AgentChipIconProps) {
  if (kind === "openhands") {
    // Logo is wider than tall — keep its native aspect ratio so it doesn't squash.
    return <OpenHandsLogo width={18} height={SIZE} className={className} />;
  }
  return <PuzzlePieceIcon width={SIZE} height={SIZE} className={className} />;
}
