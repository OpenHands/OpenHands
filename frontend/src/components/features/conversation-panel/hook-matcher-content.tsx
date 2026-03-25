import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { Pre } from "#/ui/pre";
import { HookMatcher } from "#/api/conversation-service/v1-conversation-service.types";

interface HookMatcherContentProps {
  matcher: HookMatcher;
}

export function HookMatcherContent({ matcher }: HookMatcherContentProps) {
  const { t } = useTranslation();

  return (
    <div className="mb-4 p-3 bg-gray-800 rounded-md">
      <div className="mb-2">
        <Typography.Text className="text-sm font-semibold text-gray-300">
          {t(I18nKey.HOOKS_MODAL$MATCHER)}
        </Typography.Text>
        <Typography.Text className="ml-2 px-2 py-1 text-xs rounded-full bg-blue-900">
          {matcher.matcher}
        </Typography.Text>
      </div>

      <div className="mt-3">
        <Typography.Text className="text-sm font-semibold text-gray-300 mb-2">
          {t(I18nKey.HOOKS_MODAL$COMMANDS)}
        </Typography.Text>
        {(matcher.hooks ?? []).map((hook, index) => (
          <div key={`${hook.command}-${index}`} className="mt-2">
            <Pre
              size="default"
              font="mono"
              lineHeight="relaxed"
              background="dark"
              textColor="light"
              padding="medium"
              borderRadius="medium"
              shadow="inner"
              maxHeight="small"
              overflow="auto"
            >
              {hook.command}
            </Pre>
            <div className="flex gap-4 mt-1 text-xs text-gray-400">
              <span>{t(I18nKey.HOOKS_MODAL$TYPE, { type: hook.type })}</span>
              <span>
                {t(I18nKey.HOOKS_MODAL$TIMEOUT, { timeout: hook.timeout })}
              </span>
              {hook.async ? (
                <span className="rounded-full bg-emerald-900 px-2 py-0.5 text-emerald-300">
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
