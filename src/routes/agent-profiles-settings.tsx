import { AgentProfilesLocalView } from "#/components/features/settings/agent-profiles";

export const handle = { hideTitle: false };

/**
 * Settings → Agent profiles. A library of named agent setups, reusing the
 * existing Agent settings form as the editor. Available on both local and
 * cloud backends — the cloud enterprise app-server exposes the same
 * `/api/agent-profiles` surface (OpenHands #15060, epic #3730), and
 * `AgentProfilesService` routes cloud calls through the cloud proxy. The view
 * name is historical ("LocalView"); it is backend-agnostic.
 */
export default function AgentProfilesSettingsRoute() {
  return <AgentProfilesLocalView />;
}
