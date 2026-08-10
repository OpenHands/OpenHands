/**
 * Legacy upstream update card.
 *
 * Heimdall fork: intentionally a no-op. This tree has diverged from
 * OpenHands / `@openhands/agent-canvas`; we do not poll npm for updates.
 * Interesting upstream changes are ported on request.
 */
export function AgentCanvasUpdateCard(
  _props: {
    hideWhenUpToDate?: boolean;
  } = {},
) {
  return null;
}
