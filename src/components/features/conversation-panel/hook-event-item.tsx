import { useTranslation } from "react-i18next";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Typography } from "#/ui/typography";
import { HookEvent } from "#/api/conversation-service/agent-server-conversation-service.types";
import { HookMatcherContent } from "./hook-matcher-content";
import { I18nKey } from "#/i18n/declaration";

interface HookEventItemProps {
  hookEvent: HookEvent;
  isExpanded: boolean;
  onToggle: (eventType: string) => void;
}

const EVENT_TYPE_I18N_KEYS: Record<string, I18nKey> = {
  pre_tool_use: I18nKey.HOOKS_MODAL$EVENT_PRE_TOOL_USE,
  post_tool_use: I18nKey.HOOKS_MODAL$EVENT_POST_TOOL_USE,
  user_prompt_submit: I18nKey.HOOKS_MODAL$EVENT_USER_PROMPT_SUBMIT,
  session_start: I18nKey.HOOKS_MODAL$EVENT_SESSION_START,
  session_end: I18nKey.HOOKS_MODAL$EVENT_SESSION_END,
  stop: I18nKey.HOOKS_MODAL$EVENT_STOP,
};

const HOOK_PILL_CLASS =
  "inline-flex shrink-0 items-center rounded-full px-2 py-0.5 text-[11px] font-medium leading-4 border border-[var(--oh-border)] bg-[var(--oh-surface)] text-tertiary-light";

const HOOK_COUNT_I18N_KEY = "HOOKS_MODAL$HOOK_COUNT";

export function HookEventItem({
  hookEvent,
  isExpanded,
  onToggle,
}: HookEventItemProps) {
  const { t } = useTranslation("openhands");
  const i18nKey = EVENT_TYPE_I18N_KEYS[hookEvent.event_type];
  const eventTypeLabel = i18nKey ? t(i18nKey) : hookEvent.event_type;

  const totalHooks = hookEvent.matchers.reduce(
    (sum, matcher) => sum + (matcher.hooks ?? []).length,
    0,
  );

  return (
    <div>
      <button
        type="button"
        onClick={() => onToggle(hookEvent.event_type)}
        className="w-full py-3 px-3 text-left flex items-center justify-between hover:bg-tertiary transition-colors"
      >
        <div className="flex items-center">
          <Typography.Text className="font-bold text-content-2">
            {eventTypeLabel}
          </Typography.Text>
        </div>
        <div className="flex items-center gap-2">
          <span className={HOOK_PILL_CLASS}>
            {t(HOOK_COUNT_I18N_KEY, { count: totalHooks })}
          </span>
          <Typography.Text className="text-[var(--oh-text-tertiary)]">
            {isExpanded ? (
              <ChevronDown size={18} />
            ) : (
              <ChevronRight size={18} />
            )}
          </Typography.Text>
        </div>
      </button>

      {isExpanded && (
        <div className="border-t border-[var(--oh-border)] px-3 pt-3 pb-3">
          <div className="divide-y divide-[var(--oh-border)]">
            {hookEvent.matchers.map((matcher, index) => (
              <HookMatcherContent
                key={`${hookEvent.event_type}-${matcher.matcher}-${index}`}
                matcher={matcher}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
