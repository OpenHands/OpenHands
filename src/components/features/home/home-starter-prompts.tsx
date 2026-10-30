import { useTranslation } from "react-i18next";
import { Code2, ListTodo, Sparkles, SquareTerminal } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { useConversationStore } from "#/stores/conversation-store";
import { NON_REPO_SUGGESTIONS } from "#/utils/suggestions/non-repo-suggestions";

const STARTER_PROMPT_ICONS: Partial<Record<string, typeof Code2>> = {
  [I18nKey.SUGGESTIONS$HACKER_NEWS]: SquareTerminal,
  [I18nKey.SUGGESTIONS$HELLO_WORLD]: Code2,
  [I18nKey.SUGGESTIONS$TODO_APP]: ListTodo,
};

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
      className="mt-6 flex flex-col gap-3"
    >
      <span className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-text-dim">
        <Sparkles className="size-3.5" aria-hidden />
        {label}
      </span>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {Object.entries(NON_REPO_SUGGESTIONS).map(([titleKey, prompt]) => {
          const Icon = STARTER_PROMPT_ICONS[titleKey] ?? Sparkles;
          return (
            <button
              key={titleKey}
              type="button"
              onClick={() => setMessageToSend(prompt, "home")}
              className="group flex cursor-pointer items-start gap-3 rounded-xl border border-border bg-surface-raised/40 p-3.5 text-left transition-colors hover:border-interactive-hover hover:bg-surface-raised"
            >
              <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-tertiary text-muted group-hover:text-contrast">
                <Icon className="size-4" aria-hidden />
              </span>
              <span className="text-sm leading-5 text-contrast">
                {t(titleKey as I18nKey)}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
