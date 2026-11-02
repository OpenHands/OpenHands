import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useSessionActivities } from "#/hooks/use-session-activities";
import { useSessionActivityStore } from "#/stores/session-activity-store";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { DisclosureCaret } from "#/components/shared/disclosure-caret";
import type { SessionActivity } from "#/stores/session-activity-store";
import { getTaskStatusI18nKey } from "#/utils/status";

export function sessionActivityTitle(session: SessionActivity): I18nKey {
  if (session.phase === "failed") return I18nKey.COMMON$ERROR;
  if (session.phase === "pending")
    return session.kind === "resume"
      ? I18nKey.EVENT_GROUP$SESSION_RESUMING
      : I18nKey.EVENT_GROUP$SESSION_INITIALIZING;
  return session.kind === "resume"
    ? I18nKey.EVENT_GROUP$SESSION_RESUMED
    : I18nKey.EVENT_GROUP$SESSION_INITIALIZED;
}

export function SessionActivityDetails({
  session,
}: {
  session: SessionActivity;
}) {
  const { t } = useTranslation("openhands");
  const { conversationId } = useOptionalConversationId();
  const { key } = useSessionActivities(conversationId);
  const open = session.detailsOpen ?? false;
  const setOpen = (value: boolean) =>
    useSessionActivityStore.getState().setDetailsOpen(key, session.id, value);
  const steps = session.steps.filter(
    (step) => step !== "READY" && step !== "ERROR",
  );
  return (
    <div className="py-2" data-testid="session-activity-details">
      {steps.length ? (
        <>
          <button
            type="button"
            aria-expanded={open}
            onClick={() => setOpen(!open)}
            className="flex items-center gap-2 text-muted"
          >
            <span>{t(sessionActivityTitle(session))}</span>
            <DisclosureCaret open={open} />
          </button>
          {open && (
            <ul className="mt-2 space-y-1 text-muted">
              {steps.map((step) => (
                <li key={step}>{t(getTaskStatusI18nKey(step))}</li>
              ))}
            </ul>
          )}
        </>
      ) : (
        <span className="text-muted">{t(sessionActivityTitle(session))}</span>
      )}
    </div>
  );
}
