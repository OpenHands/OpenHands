import { useTranslation } from "react-i18next";
import { MetaLlmSettingsView } from "#/components/features/settings/meta-llm-profiles";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { I18nKey } from "#/i18n/declaration";

/**
 * Settings route for managing *meta-profiles* — declarative model-routing
 * configurations consumed by the agent's ``classify_and_switch_llm`` tool.
 *
 * Local backends store meta-profiles on the agent-server. Organization-bound
 * cloud backends store them in the SaaS control plane and pass the active
 * configuration inline to ephemeral runtimes.
 *
 * Note: This is a route file, only the router should import the default export.
 */
export default function MetaLlmSettingsRoute() {
  const { t } = useTranslation("openhands");
  const { backend, orgId } = useActiveBackend();

  if (backend.kind === "cloud" && !orgId) {
    return (
      <p
        data-testid="meta-profile-cloud-unsupported"
        className="text-sm text-[var(--oh-muted)]"
      >
        {t(I18nKey.SETTINGS$META_PROFILE_CLOUD_UNSUPPORTED)}
      </p>
    );
  }

  return <MetaLlmSettingsView />;
}
