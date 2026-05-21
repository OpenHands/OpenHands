import CircuitIcon from "#/icons/u-circuit.svg?react";
import PuzzlePieceIcon from "#/icons/u-puzzle-piece.svg?react";
import type { AgentChipKind } from "#/utils/agent-display-label";

interface AgentChipIconProps {
  kind: AgentChipKind;
  className?: string;
}

/**
 * Icon for the conversation chip. The icon encodes the harness — circuit for
 * native OpenHands conversations, puzzle piece for external ACP agents —
 * leaving the chip's text free to carry the LLM/provider label.
 */
export function AgentChipIcon({
  kind,
  className = "shrink-0",
}: AgentChipIconProps) {
  const Icon = kind === "acp" ? PuzzlePieceIcon : CircuitIcon;
  return <Icon width={12} height={12} className={className} />;
}
