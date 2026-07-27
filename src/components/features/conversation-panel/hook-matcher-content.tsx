import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { Pre } from "#/ui/pre";
import { HookMatcher } from "#/api/conversation-service/agent-server-conversation-service.types";

interface HookMatcherContentProps {
  matcher: HookMatcher;
}

const HOOK_PILL_CLASS =
  "inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium leading-4 border border-[var(--oh-border)] bg-[var(--oh-surface)] text-tertiary-light";

export function HookMatcherContent({ matcher }: HookMatcherContentProps) {
  const { t } = useTranslation("openhands");

  return (
    <div className="py-3">
      <div className="mb-2">
        <Typography.Text className="text-sm font-semibold text-[var(--oh-text-tertiary)]">
          {t(I18nKey.HOOKS_MODAL$MATCHER)}
        </Typography.Text>
        <span className={`ml-2 ${HOOK_PILL_CLASS}`}>{matcher.matcher}</span>
      </div>

      <div className="mt-2">
        <Typography.Text className="text-sm font-semibold text-[var(--oh-text-tertiary)] mb-2">
          {t(I18nKey.HOOKS_MODAL$COMMANDS)}
        </Typography.Text>
        {(matcher.hooks ?? []).map((hook, index) => (
          <div key={`${hook.command}-${index}`} className="mt-2">
            <Pre
              size="small"
              font="mono"
              lineHeight="relaxed"
              padding="medium"
              borderRadius="medium"
              maxHeight="small"
              overflow="auto"
              className="border border-[var(--oh-border)] bg-base text-[var(--oh-text-tertiary)]"
            >
              {hook.command}
            </Pre>
            <div className="mt-2 flex flex-wrap gap-2 text-xs text-[var(--oh-muted)]">
              <span className={HOOK_PILL_CLASS}>
                {t(I18nKey.HOOKS_MODAL$TYPE, { type: hook.type })}
              </span>
              <span className={HOOK_PILL_CLASS}>
                {t(I18nKey.HOOKS_MODAL$TIMEOUT, { timeout: hook.timeout })}
              </span>
              {hook.async ? (
                <span className={HOOK_PILL_CLASS}>
                  {t(I18nKey.HOOKS_MODAL$ASYNC)}
                </span>
              ) : null}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
