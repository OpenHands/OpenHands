import { AgentProfilesLocalView } from "#/components/features/settings/agent-profiles";
import { settingsLikeMainScrollClassName } from "#/utils/settings-like-page-layout-classes";

export default function AgentsRoute() {
  return (
    <main data-testid="agents-page" className={settingsLikeMainScrollClassName}>
      <div className="mx-auto flex w-full min-w-0 max-w-200 flex-col gap-6">
        <AgentProfilesLocalView />
      </div>
    </main>
  );
}
