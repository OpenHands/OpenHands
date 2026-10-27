import { useState, type FormEvent } from "react";
import { useInfiniteQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useAutomationSubPageNav } from "#/components/features/automations/dashboard/use-automation-sub-page-nav";
import { ManifestSubpageLayout } from "#/components/features/manifest/manifest-subpage-layout";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { AUTOMATION_WEBHOOK_QUERY_KEYS } from "#/hooks/query/query-keys";
import { getErrorStatus } from "#/hooks/query/use-settings";
import { useAutomationPermissions } from "#/hooks/use-automation-permissions";
import { I18nKey } from "#/i18n/declaration";
import { getEventsPageSpec } from "#/manifests/automation-interface";
import { getApiErrorMessage } from "#/utils/api-error-message";

const PAGE_SIZE = 50;

export const clientLoader = () => {
  if (!getEventsPageSpec()) {
    throw new Response(null, { status: 404, statusText: "Not Found" });
  }
  return null;
};

function EventsContent({
  scope,
}: {
  scope: readonly [string, string | null, number];
}) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const {
    canView,
    canManage,
    isLoading: permissionsLoading,
  } = useAutomationPermissions();
  const [isFormOpen, setFormOpen] = useState(false);
  const [isCreating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [generatedSecret, setGeneratedSecret] = useState<string | null>(null);
  const [copyMessage, setCopyMessage] = useState<string | null>(null);
  const [suppliedSecretNotice, setSuppliedSecretNotice] = useState(false);
  const queryKey = AUTOMATION_WEBHOOK_QUERY_KEYS.forScope(...scope);
  const list = useInfiniteQuery({
    queryKey,
    queryFn: ({ pageParam }) =>
      AutomationService.listCustomWebhooks({
        limit: PAGE_SIZE,
        offset: pageParam,
      }),
    initialPageParam: 0,
    getNextPageParam: (lastPage, _pages, lastOffset) =>
      lastOffset + lastPage.webhooks.length < lastPage.total &&
      lastPage.webhooks.length > 0
        ? lastOffset + lastPage.webhooks.length
        : undefined,
    enabled: !permissionsLoading && canView,
    retry: (count, error) =>
      count < 1 && ![403, 404].includes(getErrorStatus(error) ?? 0),
    meta: { disableToast: true },
  });
  const sources = list.data?.pages.flatMap((page) => page.webhooks) ?? [];

  async function copy(value: string) {
    try {
      await navigator.clipboard.writeText(value);
      setCopyMessage(t(I18nKey.AUTOMATIONS$EVENTS$COPIED));
    } catch {
      setCopyMessage(t(I18nKey.AUTOMATIONS$EVENTS$COPY_FAILED));
    }
  }

  async function create(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canManage || isCreating) return;
    const form = event.currentTarget;
    const data = new FormData(form);
    const secret = String(data.get("webhook_secret") ?? "");
    setCreating(true);
    setCreateError(null);
    try {
      const result = await AutomationService.createCustomWebhook({
        name: String(data.get("name") ?? "").trim(),
        source: String(data.get("source") ?? "").trim(),
        event_key_expr: String(data.get("event_key_expr") ?? "type").trim(),
        signature_header: String(
          data.get("signature_header") ?? "X-Signature-256",
        ).trim(),
        ...(secret ? { webhook_secret: secret } : {}),
      });
      form.reset();
      setFormOpen(false);
      setGeneratedSecret(result.webhook_secret || null);
      setSuppliedSecretNotice(!!secret && !result.webhook_secret);
      setCopyMessage(null);
      void queryClient.invalidateQueries({ queryKey });
    } catch (error) {
      const fallback = t(I18nKey.AUTOMATIONS$EVENTS$CREATE_ERROR);
      setCreateError(
        getErrorStatus(error) === 409
          ? t(I18nKey.AUTOMATIONS$EVENTS$CONFLICT)
          : getErrorStatus(error) === 422
            ? getApiErrorMessage(error, fallback)
            : fallback,
      );
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <p
        role="note"
        className="rounded-lg border border-border p-4 text-sm text-content"
      >
        {t(I18nKey.AUTOMATIONS$EVENTS$REACHABILITY_WARNING)}
      </p>
      {generatedSecret && (
        <section role="alert" className="rounded-lg border border-border p-4">
          <h2 className="font-semibold">
            {t(I18nKey.AUTOMATIONS$EVENTS$SECRET_TITLE)}
          </h2>
          <p className="text-sm">{t(I18nKey.AUTOMATIONS$EVENTS$SECRET_ONCE)}</p>
          <code className="block break-all py-2">{generatedSecret}</code>
          <div className="flex gap-4">
            <button type="button" onClick={() => copy(generatedSecret)}>
              {t(I18nKey.BUTTON$COPY)}
            </button>
            <button
              type="button"
              onClick={() => {
                setGeneratedSecret(null);
                setCopyMessage(null);
              }}
            >
              {t(I18nKey.AUTOMATIONS$EVENTS$DONE)}
            </button>
          </div>
          {copyMessage && <p role="status">{copyMessage}</p>}
        </section>
      )}
      {permissionsLoading || (canView && list.isPending) ? (
        <p role="status">{t(I18nKey.AUTOMATIONS$EVENTS$LOADING)}</p>
      ) : !canView || getErrorStatus(list.error) === 403 ? (
        <p role="alert">{t(I18nKey.AUTOMATIONS$EVENTS$FETCH_ERROR)}</p>
      ) : getErrorStatus(list.error) === 404 ? (
        <p role="alert">{t(I18nKey.AUTOMATIONS$EVENTS$UNSUPPORTED)}</p>
      ) : list.isError && !list.data ? (
        <div>
          <p role="alert">{t(I18nKey.AUTOMATIONS$EVENTS$FETCH_ERROR)}</p>
          <button type="button" onClick={() => void list.refetch()}>
            {t(I18nKey.AUTOMATIONS$ERROR_RETRY)}
          </button>
        </div>
      ) : (
        <>
          {list.isError && (
            <div>
              <p role="alert">{t(I18nKey.AUTOMATIONS$EVENTS$FETCH_ERROR)}</p>
              <button type="button" onClick={() => void list.refetch()}>
                {t(I18nKey.AUTOMATIONS$ERROR_RETRY)}
              </button>
            </div>
          )}
          {canManage && (
            <div>
              <button
                type="button"
                className="rounded-md bg-primary px-4 py-2 text-sm text-on-primary"
                onClick={() => {
                  setFormOpen((open) => !open);
                  setCreateError(null);
                }}
              >
                {t(
                  isFormOpen
                    ? I18nKey.BUTTON$CANCEL
                    : I18nKey.AUTOMATIONS$EVENTS$ADD,
                )}
              </button>
            </div>
          )}
          {canManage && isFormOpen && (
            <form
              onSubmit={create}
              autoComplete="off"
              className="flex flex-col gap-4 rounded-lg border border-border p-4"
            >
              <label className="flex flex-col gap-1 text-sm">
                {t(I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_NAME)}
                <input
                  name="name"
                  required
                  maxLength={500}
                  className="rounded border border-border p-2"
                />
              </label>
              <label className="flex flex-col gap-1 text-sm">
                {t(I18nKey.AUTOMATIONS$EVENTS$SOURCE)}
                <input
                  name="source"
                  required
                  pattern="[a-z0-9]([a-z0-9-]{0,48}[a-z0-9])?"
                  maxLength={50}
                  className="rounded border border-border p-2"
                />
                <span className="text-muted">
                  {t(I18nKey.AUTOMATIONS$EVENTS$SOURCE_HELP)}
                </span>
              </label>
              <label className="flex flex-col gap-1 text-sm">
                {t(I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_EVENT_KEY_EXPR)}
                <input
                  name="event_key_expr"
                  defaultValue="type"
                  required
                  maxLength={500}
                  className="rounded border border-border p-2"
                />
              </label>
              <label className="flex flex-col gap-1 text-sm">
                {t(I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_SIGNATURE_HEADER)}
                <input
                  name="signature_header"
                  defaultValue="X-Signature-256"
                  required
                  maxLength={100}
                  className="rounded border border-border p-2"
                />
              </label>
              <label className="flex flex-col gap-1 text-sm">
                {t(I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_SECRET)}
                <input
                  name="webhook_secret"
                  type="password"
                  minLength={8}
                  maxLength={255}
                  autoComplete="new-password"
                  className="rounded border border-border p-2"
                />
                <span className="text-muted">
                  {t(I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_SECRET_HELP)}
                </span>
              </label>
              {createError && (
                <p role="alert" className="text-status-error">
                  {createError}
                </p>
              )}
              <button
                type="submit"
                disabled={isCreating}
                className="self-start rounded-md bg-primary px-4 py-2 text-sm text-on-primary disabled:opacity-50"
              >
                {t(I18nKey.AUTOMATIONS$EVENTS$ADD)}
              </button>
            </form>
          )}
          {suppliedSecretNotice && (
            <p role="status">
              {t(I18nKey.AUTOMATIONS$EVENTS$CREATED_WITH_OWN_SECRET)}
            </p>
          )}
          {sources.length === 0 ? (
            <p className="text-sm text-muted">
              {t(I18nKey.AUTOMATIONS$EVENTS$EMPTY)}
            </p>
          ) : (
            <ul className="flex flex-col gap-3">
              {sources.map((source) => (
                <li
                  key={source.id}
                  className="rounded-lg border border-border p-4"
                >
                  <div className="font-semibold text-content">
                    {source.name}
                  </div>
                  <div className="text-sm text-muted">
                    {source.source} ·{" "}
                    {source.enabled
                      ? t(I18nKey.AUTOMATIONS$ACTIVE)
                      : t(I18nKey.AUTOMATIONS$INACTIVE)}
                  </div>
                  <code className="block break-all py-2 text-sm">
                    {source.webhook_url}
                  </code>
                  <button
                    type="button"
                    className="text-sm underline"
                    onClick={() => copy(source.webhook_url)}
                  >
                    {t(I18nKey.AUTOMATIONS$EVENTS$COPY_URL)}
                  </button>
                </li>
              ))}
            </ul>
          )}
          {list.hasNextPage && (
            <button
              type="button"
              disabled={list.isFetchingNextPage}
              className="self-start text-sm underline"
              onClick={() => list.fetchNextPage()}
            >
              {t(I18nKey.AUTOMATIONS$EVENTS$LOAD_MORE)}
            </button>
          )}
          {copyMessage && !generatedSecret && (
            <p role="status">{copyMessage}</p>
          )}
        </>
      )}
    </div>
  );
}

export default function AutomationEvents() {
  const { backend, orgId } = useActiveBackend();
  const spec = getEventsPageSpec();
  const nav = useAutomationSubPageNav();
  if (!spec || !nav) return null;
  return (
    <ManifestSubpageLayout
      heading={nav.heading}
      navTestIdBase="automations-navbar"
      items={nav.items}
    >
      <div>
        <h1 className="text-xl font-semibold text-content">{spec.title}</h1>
        <p className="mt-1 text-sm text-muted">{spec.description}</p>
      </div>
      <EventsContent
        key={`${backend.id}:${orgId}:${backend.connectionRevision ?? 0}`}
        scope={[backend.id, orgId, backend.connectionRevision ?? 0]}
      />
    </ManifestSubpageLayout>
  );
}
