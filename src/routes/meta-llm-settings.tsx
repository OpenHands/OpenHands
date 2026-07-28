import { useTranslation } from "react-i18next";
import { MetaLlmSettingsView } from "#/components/features/settings/meta-llm-profiles";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { I18nKey } from "#/i18n/declaration";

/**
 * Settings route for managing *meta-profiles* — declarative model-routing
 * configurations consumed by the agent's ``classify_and_switch_llm`` tool.
 *
 * Meta-profiles (like LLM profiles) are stored on the local agent-server, so
 * the management view is only available for local backends. Cloud backends get
 * an explanatory message.
 *
 * Note: This is a route file, only the router should import the default export.
 */
export default function MetaLlmSettingsRoute() {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();

  if (backend.kind === "cloud") {
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
