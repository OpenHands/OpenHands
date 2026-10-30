import { useTranslation } from "react-i18next";
import { Sparkles } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { useConversationStore } from "#/stores/conversation-store";
import { NON_REPO_SUGGESTIONS } from "#/utils/suggestions/non-repo-suggestions";

/**
 * Example prompts under the Home composer for people who do not yet know what
 * to ask. Choosing one fills the composer (it does not send), so the user can
 * read and edit the full prompt first.
 */
export function HomeStarterPrompts() {
  const { t } = useTranslation("openhands");
  const { setMessageToSend } = useConversationStore();
  const label = t(I18nKey.HOME$STARTER_PROMPTS_LABEL);

  return (
    <div
      role="group"
      aria-label={label}
      data-testid="home-starter-prompts"
      className="flex flex-col gap-2"
    >
      <span className="flex items-center gap-1.5 text-xs text-text-secondary">
        <Sparkles className="size-3.5" aria-hidden />
        {label}
      </span>
      <div className="flex flex-wrap gap-2">
        {Object.entries(NON_REPO_SUGGESTIONS).map(([titleKey, prompt]) => (
          <button
            key={titleKey}
            type="button"
            onClick={() => setMessageToSend(prompt, "home")}
            className="cursor-pointer rounded-full border border-border px-3 py-1.5 text-left text-sm text-contrast hover:border-interactive-hover hover:bg-surface-raised"
          >
            {t(titleKey as I18nKey)}
          </button>
        ))}
      </div>
    </div>
  );
}
