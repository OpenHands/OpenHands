import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { I18nKey } from "#/i18n/declaration";
import { useAcpPermissions } from "#/hooks/query/use-acp-permissions";

export function AcpPermissionPanel({
  conversation,
}: {
  conversation: AppConversation | null | undefined;
}) {
  const { t } = useTranslation();
  const { pending, decision, scope, enabled } = useAcpPermissions(conversation);
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    if (!enabled) return undefined;
    const timer = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(timer);
  }, [enabled]);
  if (!enabled) return null;
  const error =
    pending.isError ||
    (decision.isError && decision.variables?.requestScope === scope);
  return (
    <section
      aria-label={t(I18nKey.ACP_PERMISSION_REQUIRED)}
      className="flex flex-col gap-2"
    >
      {error && (
        <div role="alert">
          {t(I18nKey.ERROR$GENERIC)}{" "}
          <button
            type="button"
            onClick={() => {
              decision.reset();
              void pending.refetch();
            }}
          >
            {t(I18nKey.BUTTON$REFRESH)}
          </button>
        </div>
      )}
      {pending.data?.items.map((request) => {
        const expires = Date.parse(request.expires_at);
        const expired = !Number.isFinite(expires) || expires <= now;
        const busy =
          decision.isPending && decision.variables?.requestScope === scope;
        const decide = (optionId: string | null) =>
          decision.mutate({
            requestId: request.request_id,
            optionId,
            requestScope: scope,
          });
        return (
          <article key={request.request_id} className="rounded border p-3">
            <h2>{t(I18nKey.ACP_PERMISSION_REQUIRED)}</h2>
            <p>{request.title}</p>
            <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words">
              {request.details}
            </pre>
            {request.details_truncated && (
              <p role="alert">{t(I18nKey.ACP_PERMISSION_INCOMPLETE)}</p>
            )}
            <div className="flex flex-wrap gap-2">
              {request.options.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  disabled={
                    busy ||
                    expired ||
                    pending.isError ||
                    (request.details_truncated &&
                      option.kind.startsWith("allow"))
                  }
                  onClick={() => decide(option.id)}
                >
                  {option.name}
                </button>
              ))}
              <button
                type="button"
                disabled={busy || expired}
                onClick={() => decide(null)}
              >
                {t(I18nKey.BUTTON$CANCEL)}
              </button>
            </div>
          </article>
        );
      })}
    </section>
  );
}
