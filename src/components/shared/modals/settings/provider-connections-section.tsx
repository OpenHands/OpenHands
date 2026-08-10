/**
 * Provider Connections section (scaffold, draft).
 *
 * Shown on the LLM Profiles page:
 *   Anthropic - 6 models - last refreshed 3 days ago
 *     -> Refresh / Rotate key / Disconnect
 *
 * Rotate re-saves the named secret; Disconnect removes the connection
 * (optionally its spawned profiles).
 *
 * Tracking: OpenHands/OpenHands#15492, Linear OSS-5295.
 * Scope: OpenHands frontend PR4 of the provider-connections plan.
 *
 * TODO (implementation):
 *  - Section component listing connections + actions.
 *  - Refresh = re-validate + re-fetch catalog (pull-on-demand, no bg job).
 *  - Rotate = update the named secret the connection/profiles reference.
 *  - Disconnect = delete connection; decide profile fate (keep/delete).
 *  - Decide behavior for running conversations using a rotated/disconnected key.
 */

export function ProviderConnectionsSection() {
  // TODO
  return null;
}
