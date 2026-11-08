import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import EventService from "#/api/event-service/event-service.api";
import type { ResolvedActiveBackend } from "#/api/backend-registry/types";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { assertActiveEnvironment } from "#/utils/assert-active-environment";
import { I18nKey } from "#/i18n/declaration";
import {
  HistorySearchSession,
  historyEventText,
  loadHistoryContext,
  type HistoryMatch,
  type HistorySearchSnapshot,
} from "#/utils/conversation-history-search";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import { isMessageEvent } from "#/types/agent-server/type-guards";

interface HistorySearchProps {
  conversationId: string;
  conversationUrl?: string | null;
  sessionApiKey?: string | null;
}

const buttonClass =
  "rounded-md border border-border px-3 py-1.5 text-sm text-contrast hover:bg-surface-raised disabled:opacity-50";

/** Kept separate from live chat: no event-store replacement or agent execution. */
export function ConversationHistorySearch(props: HistorySearchProps) {
  const { t } = useTranslation("openhands");
  const environment = useActiveBackend();
  const [open, setOpen] = useState(false);
  const trigger = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    setOpen(false);
  }, [props.sessionApiKey]);
  const scope = JSON.stringify([
    environment.backend.id,
    environment.backend.connectionRevision,
    environment.orgId,
    props.conversationId,
    props.conversationUrl,
  ]);
  const close = () => {
    setOpen(false);
    trigger.current?.focus();
  };
  return (
    <div className="flex shrink-0 justify-end px-3 py-1">
      <button
        ref={trigger}
        type="button"
        className="text-xs text-muted underline hover:text-contrast"
        onClick={() => setOpen(true)}
      >
        {t(I18nKey.HISTORY_SEARCH$OPEN)}
      </button>
      {open && (
        <HistorySearchDialog
          key={scope}
          {...props}
          environment={environment}
          onClose={close}
        />
      )}
    </div>
  );
}

function HistorySearchDialog({
  conversationId,
  conversationUrl,
  sessionApiKey,
  environment,
  onClose,
}: HistorySearchProps & {
  environment: ResolvedActiveBackend;
  onClose: () => void;
}) {
  const { t } = useTranslation("openhands");
  const [query, setQuery] = useState("");
  const [searchedQuery, setSearchedQuery] = useState("");
  const [snapshot, setSnapshot] = useState<HistorySearchSnapshot | null>(null);
  const [preparing, setPreparing] = useState(false);
  const [context, setContext] = useState<{
    match: HistoryMatch;
    events: OpenHandsEvent[];
  } | null>(null);
  const [contextLoading, setContextLoading] = useState(false);
  const [contextError, setContextError] = useState(false);
  const session = useRef<HistorySearchSession | null>(null);
  const version = useRef(0);
  const contextVersion = useRef(0);
  const mounted = useRef(true);
  const scanning = useRef(false);
  const resultButtons = useRef<(HTMLButtonElement | null)[]>([]);
  const contextHeading = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      version.current += 1;
      contextVersion.current += 1;
      session.current?.cancel();
    };
  }, []);

  const searchPage = async (
    options: Parameters<typeof EventService.searchEvents>[3],
  ) => {
    assertActiveEnvironment(environment);
    const result = await EventService.searchEvents(
      conversationId,
      conversationUrl,
      sessionApiKey,
      options,
    );
    assertActiveEnvironment(environment);
    return result;
  };

  const scan = async (fresh: boolean) => {
    if (scanning.current) return;
    scanning.current = true;
    const request = ++version.current;
    const current = () => mounted.current && version.current === request;
    if (fresh) {
      session.current?.cancel();
      contextVersion.current += 1;
      setContext(null);
      setContextError(false);
      setSnapshot(null);
      setSearchedQuery(query.trim());
      setPreparing(true);
      let total: number | undefined;
      try {
        assertActiveEnvironment(environment);
        // Cloud persisted history remains searchable when its runtime has stopped.
        if (conversationUrl || environment.backend.kind === "local")
          total = await EventService.getEventCount(
            conversationId,
            conversationUrl ?? "",
            sessionApiKey,
          );
      } catch {
        /* Count is optional; cursor pagination still proves exhaustion. */
      }
      if (!current()) return;
      setPreparing(false);
      session.current = new HistorySearchSession(query, total);
    }
    if (!current() || !session.current) return;
    await session.current.scanBatch(searchPage, (next) => {
      if (current()) setSnapshot(next);
    });
    if (current()) scanning.current = false;
  };

  const cancel = () => {
    version.current += 1;
    scanning.current = false;
    setPreparing(false);
    const stopped = session.current?.cancel();
    setSnapshot(
      stopped
        ? { ...stopped, phase: "cancelled", running: false }
        : {
            matches: [],
            examined: 0,
            phase: "cancelled",
            running: false,
            limitReached: false,
            error: null,
          },
    );
  };

  const openResult = async (match: HistoryMatch) => {
    const request = ++contextVersion.current;
    setContext(null);
    setContextLoading(true);
    setContextError(false);
    try {
      const events = await loadHistoryContext(match, searchPage);
      if (mounted.current && request === contextVersion.current)
        setContext({ match, events });
    } catch {
      if (mounted.current && request === contextVersion.current)
        setContextError(true);
    } finally {
      if (mounted.current && request === contextVersion.current)
        setContextLoading(false);
    }
  };
  useEffect(() => {
    if (context) contextHeading.current?.focus();
  }, [context]);

  const moveResult = (
    event: KeyboardEvent<HTMLButtonElement>,
    index: number,
  ) => {
    let next: number;
    if (event.key === "ArrowDown")
      next = (index + 1) % (snapshot?.matches.length ?? 1);
    else if (event.key === "ArrowUp")
      next =
        (index - 1 + (snapshot?.matches.length ?? 1)) %
        (snapshot?.matches.length ?? 1);
    else return;
    event.preventDefault();
    resultButtons.current[next]?.focus();
  };
  const busy = preparing || snapshot?.running;
  const phase = snapshot?.phase;
  const target = context?.events.find((event) => event.id === context.match.id);
  const relatedToolId =
    target && "tool_call_id" in target ? target.tool_call_id : undefined;
  const statusKey =
    phase === "complete"
      ? I18nKey.HISTORY_SEARCH$COMPLETE
      : phase === "cancelled"
        ? I18nKey.HISTORY_SEARCH$CANCELLED
        : phase === "scanning"
          ? I18nKey.HISTORY_SEARCH$SCANNING
          : I18nKey.HISTORY_SEARCH$INCOMPLETE;
  return (
    <ModalBackdrop
      onClose={onClose}
      aria-label={t(I18nKey.HISTORY_SEARCH$OPEN)}
    >
      <ModalBody
        width="xl"
        className="dialog-viewport-width dialog-content-height items-stretch gap-3 overflow-y-auto"
      >
        <ModalCloseButton onClose={onClose} />
        <h2 className="pr-8 text-lg font-medium text-contrast">
          {t(I18nKey.HISTORY_SEARCH$OPEN)}
        </h2>
        <p className="text-xs text-muted">
          {t(I18nKey.HISTORY_SEARCH$DESCRIPTION)}
        </p>
        <form
          className="flex gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            if (query.trim()) void scan(true);
          }}
        >
          <input
            type="search"
            aria-label={t(I18nKey.HISTORY_SEARCH$QUERY)}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            className="min-w-0 flex-1 rounded-md border border-border bg-canvas-base px-3 py-2 text-contrast"
          />
          <button
            className={buttonClass}
            disabled={!!busy || !query.trim()}
            type="submit"
          >
            {t(I18nKey.HISTORY_SEARCH$SUBMIT)}
          </button>
        </form>
        <div role="status" aria-live="polite" className="text-sm text-muted">
          {preparing && t(I18nKey.HISTORY_SEARCH$SCANNING)}
          {snapshot && (
            <>
              <p>
                {t(I18nKey.HISTORY_SEARCH$QUERY)}: <q>{searchedQuery}</q>
              </p>
              <p>{t(statusKey)}</p>
              <p>
                {t(I18nKey.HISTORY_SEARCH$PROGRESS, {
                  examined: snapshot.examined,
                  matches: snapshot.matches.length,
                })}
              </p>
            </>
          )}
          {snapshot?.limitReached && <p>{t(I18nKey.HISTORY_SEARCH$LIMIT)}</p>}
        </div>
        {phase === "error" && (
          <p role="alert" className="text-sm text-danger">
            {t(I18nKey.HISTORY_SEARCH$ERROR)}
          </p>
        )}
        <div className="flex flex-wrap gap-2">
          {busy && (
            <button className={buttonClass} type="button" onClick={cancel}>
              {t(I18nKey.HISTORY_SEARCH$CANCEL)}
            </button>
          )}
          {!busy &&
            snapshot &&
            !snapshot.limitReached &&
            phase !== "complete" && (
              <button
                className={buttonClass}
                type="button"
                onClick={() => void scan(!session.current)}
              >
                {t(I18nKey.HISTORY_SEARCH$CONTINUE)}
              </button>
            )}
          <button className={buttonClass} type="button" onClick={onClose}>
            {t(I18nKey.HISTORY_SEARCH$RETURN_LIVE)}
          </button>
        </div>
        {!!snapshot?.matches.length && (
          <ol className="flex max-h-56 flex-col gap-2 overflow-y-auto">
            {snapshot.matches.map((match, index) => (
              <li key={match.id}>
                <button
                  ref={(element) => {
                    resultButtons.current[index] = element;
                  }}
                  type="button"
                  onKeyDown={(event) => moveResult(event, index)}
                  onClick={() => void openResult(match)}
                  className="w-full rounded-md border border-border p-2 text-start text-sm break-words hover:bg-surface-raised"
                >
                  <time
                    className="block text-xs text-muted"
                    dateTime={match.timestamp ?? undefined}
                  >
                    {match.timestamp
                      ? new Date(match.timestamp).toLocaleString()
                      : ""}
                  </time>
                  <span>{match.excerpt}</span>
                </button>
              </li>
            ))}
          </ol>
        )}
        {contextLoading && <p role="status">{t(I18nKey.HOME$LOADING)}</p>}
        {contextError && (
          <p role="alert">{t(I18nKey.HISTORY_SEARCH$CONTEXT_ERROR)}</p>
        )}
        {context && (
          <section
            data-testid="history-search-context"
            className="flex flex-col gap-2 border-t border-border pt-3"
          >
            <h3
              ref={contextHeading}
              tabIndex={-1}
              className="text-sm font-medium"
            >
              {t(I18nKey.HISTORY_SEARCH$CONTEXT)}
            </h3>
            {context.events
              .filter((event) => historyEventText(event))
              .map((event) => (
                <details
                  open={
                    event.id === context.match.id ||
                    (!!relatedToolId &&
                      "tool_call_id" in event &&
                      event.tool_call_id === relatedToolId)
                  }
                  key={event.id}
                  className="rounded-md border border-border p-2"
                  aria-current={
                    event.id === context.match.id ? "true" : undefined
                  }
                >
                  <summary className="cursor-pointer text-xs text-muted">
                    {t(
                      isMessageEvent(event)
                        ? event.llm_message.role === "user"
                          ? I18nKey.TRANSCRIPT_EXPORT$USER
                          : I18nKey.CHAT_INTERFACE$ASSISTANT
                        : I18nKey.TRANSCRIPT_EXPORT$TOOL,
                    )}{" "}
                    ·{" "}
                    <time dateTime={event.timestamp}>
                      {event.timestamp
                        ? new Date(event.timestamp).toLocaleString()
                        : ""}
                    </time>
                  </summary>
                  <pre className="mt-2 max-h-80 overflow-auto text-sm whitespace-pre-wrap break-words">
                    {historyEventText(event)}
                  </pre>
                </details>
              ))}
          </section>
        )}
      </ModalBody>
    </ModalBackdrop>
  );
}
