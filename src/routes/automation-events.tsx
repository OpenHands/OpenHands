import { useState, type FormEvent } from "react";
import { useTranslation } from "react-i18next";
import { useAutomationSubPageNav } from "#/components/features/automations/dashboard/use-automation-sub-page-nav";
import { ManifestSubpageLayout } from "#/components/features/manifest/manifest-subpage-layout";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useAutomationPermissions } from "#/hooks/use-automation-permissions";
import {
  useAutomationWebhooks,
  useCreateAutomationWebhook,
} from "#/hooks/query/use-automation-webhooks";
import { useDeploymentCapabilities } from "#/hooks/query/use-manifest-capabilities";
import { getErrorStatus } from "#/hooks/query/use-settings";
import { I18nKey } from "#/i18n/declaration";
import { getEventsPageSpec } from "#/manifests/automation-interface";
import type { CreateAutomationWebhookRequest } from "#/types/automation-webhook";

const SOURCE_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const SIGNATURE_HEADER_PATTERN = /^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/;
const EVENT_KEY_DEFAULT = "type";
const SIGNATURE_HEADER_DEFAULT = "X-Signature-256";
const SIGNATURE_SCHEME_DEFAULT = "hmac_sha256_hex";

export const clientLoader = () => {
  if (!getEventsPageSpec()) {
    throw new Response(null, { status: 404, statusText: "Not Found" });
  }
  return null;
};

// @spec BM-002 — Custom event sources
export default function AutomationEvents() {
  const { backend, orgId } = useActiveBackend();
  const scope = `${backend.id}:${backend.connectionRevision ?? 0}:${orgId ?? ""}`;
  return <EventSourcesForScope key={scope} scope={scope} />;
}

function EventSourcesForScope({ scope }: { scope: string }) {
  const { t } = useTranslation("openhands");
  const { canManage, isLoading: permissionsLoading } =
    useAutomationPermissions();
  const nav = useAutomationSubPageNav();
  const spec = getEventsPageSpec();
  const list = useAutomationWebhooks(canManage);
  const capabilities = useDeploymentCapabilities(canManage);
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState("");
  const [source, setSource] = useState("");
  const [eventKey, setEventKey] = useState(EVENT_KEY_DEFAULT);
  const [signatureHeader, setSignatureHeader] = useState(
    SIGNATURE_HEADER_DEFAULT,
  );
  const [secret, setSecret] = useState("");
  const [confirmation, setConfirmation] = useState<{
    scope: string;
    value: string;
  } | null>(null);
  const [copyMessage, setCopyMessage] = useState("");
  const create = useCreateAutomationWebhook((value) =>
    setConfirmation({ scope, value }),
  );

  if (!spec || !nav) return null;

  const copy = async (value: string) => {
    try {
      await navigator.clipboard.writeText(value);
      setCopyMessage(t(I18nKey.AUTOMATIONS$EVENTS$COPIED));
    } catch {
      setCopyMessage(t(I18nKey.AUTOMATIONS$EVENTS$COPY_FAILED));
    }
  };

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmedName = name.trim();
    const trimmedSource = source.trim();
    if (
      !trimmedName ||
      !SOURCE_PATTERN.test(trimmedSource) ||
      !eventKey.trim() ||
      !SIGNATURE_HEADER_PATTERN.test(signatureHeader.trim()) ||
      (secret.length > 0 && secret.length < 8)
    )
      return;
    const request: CreateAutomationWebhookRequest = {
      name: trimmedName,
      source: trimmedSource,
      event_key_expr: eventKey.trim(),
      signature_header: signatureHeader.trim(),
      signature_scheme: SIGNATURE_SCHEME_DEFAULT,
      ...(secret ? { webhook_secret: secret } : {}),
    };
    setSecret("");
    create.create(request);
  };

  const pages = list.data?.pages ?? [];
  const records = pages.flatMap((page) => page.webhooks);
  const unsupportedDelivery =
    capabilities.data && !capabilities.data.triggerKinds.includes("event");
  const buttonClass =
    "rounded-lg border border-border px-3 py-2 text-sm text-content hover:bg-surface-raised disabled:opacity-50";
  const inputClass =
    "w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-content";

  return (
    <ManifestSubpageLayout
      heading={nav.heading}
      navTestIdBase="automations-navbar"
      items={nav.items}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-content">{spec.title}</h1>
          <p className="mt-1 text-sm text-muted">{spec.description}</p>
        </div>
        {!permissionsLoading && canManage && list.data && (
          <button
            type="button"
            className={buttonClass}
            disabled={create.isPending}
            onClick={() => {
              setConfirmation(null);
              setAdding(true);
              create.reset();
            }}
          >
            {t(I18nKey.AUTOMATIONS$EVENTS$ADD)}
          </button>
        )}
      </div>

      {permissionsLoading ? (
        <p role="status" className="text-sm text-muted">
          {t(I18nKey.AUTOMATIONS$EVENTS$LOADING)}
        </p>
      ) : !canManage ? (
        <p role="alert" className="text-sm text-muted">
          {t(I18nKey.AUTOMATIONS$EVENTS$NO_ACCESS)}
        </p>
      ) : list.isPending ? (
        <p role="status" className="text-sm text-muted">
          {t(I18nKey.AUTOMATIONS$EVENTS$LOADING)}
        </p>
      ) : !list.data && getErrorStatus(list.error) === 404 ? (
        <p role="alert" className="text-sm text-muted">
          {t(I18nKey.AUTOMATIONS$EVENTS$UNSUPPORTED)}
        </p>
      ) : !list.data && getErrorStatus(list.error) === 403 ? (
        <p role="alert" className="text-sm text-muted">
          {t(I18nKey.AUTOMATIONS$EVENTS$NO_ACCESS)}
        </p>
      ) : !list.data ? (
        <div
          role="alert"
          className="flex flex-wrap items-center gap-3 text-sm text-muted"
        >
          <span>{t(I18nKey.AUTOMATIONS$EVENTS$FETCH_ERROR)}</span>
          <button
            type="button"
            className={buttonClass}
            onClick={() => list.refetch()}
          >
            {t(I18nKey.AUTOMATIONS$ERROR_RETRY)}
          </button>
        </div>
      ) : (
        <>
          {unsupportedDelivery && (
            <p
              role="status"
              className="rounded-lg border border-warning p-4 text-sm text-content"
            >
              {t(I18nKey.AUTOMATIONS$EVENTS$DELIVERY_UNSUPPORTED)}
            </p>
          )}
          {records.length === 0 ? (
            <div className="rounded-xl border border-border bg-surface p-5">
              <p className="font-medium text-content">
                {t(I18nKey.AUTOMATIONS$EVENTS$EMPTY)}
              </p>
              <p className="mt-2 text-sm text-muted">
                {t(I18nKey.AUTOMATIONS$EVENTS$REACHABILITY_WARNING)}
              </p>
            </div>
          ) : (
            <div className="flex flex-col gap-3">
              {records.map((webhook) => (
                <section
                  key={webhook.id}
                  className="min-w-0 rounded-xl border border-border bg-surface p-4"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <h2 className="font-medium text-content">{webhook.name}</h2>
                    <span className="text-sm text-muted">
                      {t(
                        webhook.enabled
                          ? I18nKey.AUTOMATIONS$ACTIVE
                          : I18nKey.AUTOMATIONS$INACTIVE,
                      )}
                    </span>
                  </div>
                  <p className="mt-1 text-sm text-muted">{webhook.source}</p>
                  <div className="mt-3 flex min-w-0 flex-wrap items-center gap-2">
                    <code className="min-w-0 break-all text-xs text-content">
                      {webhook.webhook_url}
                    </code>
                    <button
                      type="button"
                      className={buttonClass}
                      onClick={() => copy(webhook.webhook_url)}
                      aria-label={t(I18nKey.AUTOMATIONS$EVENTS$COPY_URL)}
                    >
                      {t(I18nKey.BUTTON$COPY)}
                    </button>
                  </div>
                </section>
              ))}
              {list.hasNextPage && (
                <button
                  type="button"
                  className={buttonClass}
                  disabled={list.isFetchingNextPage}
                  onClick={() => list.fetchNextPage()}
                >
                  {t(I18nKey.AUTOMATIONS$EVENTS$LOAD_MORE)}
                </button>
              )}
              {list.isFetchNextPageError && (
                <button
                  type="button"
                  className={buttonClass}
                  onClick={() => list.fetchNextPage()}
                >
                  {t(I18nKey.AUTOMATIONS$ERROR_RETRY)}
                </button>
              )}
            </div>
          )}

          {adding && confirmation?.scope === scope ? (
            <section
              role="status"
              className="rounded-xl border border-border bg-surface p-5"
            >
              <h2 className="font-medium text-content">
                {t(I18nKey.AUTOMATIONS$EVENTS$SECRET_TITLE)}
              </h2>
              <p className="my-2 text-sm text-muted">
                {t(I18nKey.AUTOMATIONS$EVENTS$SECRET_ONCE)}
              </p>
              <code className="block break-all text-sm text-content">
                {confirmation.value}
              </code>
              <div className="mt-3 flex gap-2">
                <button
                  type="button"
                  className={buttonClass}
                  onClick={() => copy(confirmation.value)}
                >
                  {t(I18nKey.BUTTON$COPY)}
                </button>
                <button
                  type="button"
                  className={buttonClass}
                  onClick={() => {
                    setConfirmation(null);
                    setAdding(false);
                  }}
                >
                  {t(I18nKey.AUTOMATIONS$EVENTS$DONE)}
                </button>
              </div>
            </section>
          ) : adding && create.isSuccess ? (
            <div
              role="status"
              className="rounded-xl border border-border bg-surface p-5"
            >
              <p className="text-sm text-content">
                {t(I18nKey.AUTOMATIONS$EVENTS$CREATED_WITH_OWN_SECRET)}
              </p>
              <button
                type="button"
                className={buttonClass}
                onClick={() => setAdding(false)}
              >
                {t(I18nKey.AUTOMATIONS$EVENTS$DONE)}
              </button>
            </div>
          ) : (
            adding && (
              <form
                onSubmit={submit}
                className="flex flex-col gap-4 rounded-xl border border-border bg-surface p-5"
              >
                <h2 className="font-medium text-content">
                  {t(I18nKey.AUTOMATIONS$EVENTS$ADD)}
                </h2>
                <label className="flex flex-col gap-1 text-sm text-content">
                  {t(I18nKey.AUTOMATIONS$NAME)}
                  <input
                    className={inputClass}
                    required
                    maxLength={255}
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                  />
                </label>
                <label className="flex flex-col gap-1 text-sm text-content">
                  {t(I18nKey.AUTOMATIONS$EVENTS$SOURCE)}
                  <input
                    className={inputClass}
                    required
                    maxLength={50}
                    pattern="[a-z0-9]+(-[a-z0-9]+)*"
                    value={source}
                    onChange={(e) => setSource(e.target.value)}
                  />
                  <span className="text-xs text-muted">
                    {t(I18nKey.AUTOMATIONS$EVENTS$SOURCE_HELP)}
                  </span>
                </label>
                <details className="text-sm text-content">
                  <summary className="cursor-pointer">
                    {t(I18nKey.COMMON$ADVANCED_SETTINGS)}
                  </summary>
                  <div className="mt-3 flex flex-col gap-4">
                    <label className="flex flex-col gap-1">
                      {t(I18nKey.AUTOMATIONS$EVENTS$EVENT_KEY)}
                      <input
                        className={inputClass}
                        required
                        maxLength={500}
                        value={eventKey}
                        onChange={(e) => setEventKey(e.target.value)}
                      />
                    </label>
                    <label className="flex flex-col gap-1">
                      {t(I18nKey.AUTOMATIONS$EVENTS$SIGNATURE_HEADER)}
                      <input
                        className={inputClass}
                        required
                        maxLength={100}
                        value={signatureHeader}
                        onChange={(e) => setSignatureHeader(e.target.value)}
                      />
                    </label>
                    <label className="flex flex-col gap-1">
                      {t(I18nKey.AUTOMATIONS$EVENTS$OPTIONAL_SECRET)}
                      <input
                        type="password"
                        autoComplete="new-password"
                        className={inputClass}
                        minLength={8}
                        maxLength={255}
                        value={secret}
                        onChange={(e) => setSecret(e.target.value)}
                      />
                    </label>
                    <p className="text-xs text-muted">
                      {t(I18nKey.AUTOMATIONS$EVENTS$SIGNATURE_HELP)}
                    </p>
                  </div>
                </details>
                {create.isError && (
                  <p role="alert" className="text-sm text-danger">
                    {t(
                      getErrorStatus(create.error) === 409
                        ? I18nKey.AUTOMATIONS$EVENTS$CONFLICT
                        : I18nKey.AUTOMATIONS$EVENTS$CREATE_ERROR,
                    )}
                  </p>
                )}
                <div className="flex gap-2">
                  <button
                    type="submit"
                    disabled={create.isPending}
                    className={buttonClass}
                  >
                    {t(I18nKey.BUTTON$CREATE)}
                  </button>
                  <button
                    type="button"
                    disabled={create.isPending}
                    className={buttonClass}
                    onClick={() => {
                      setSecret("");
                      setAdding(false);
                    }}
                  >
                    {t(I18nKey.BUTTON$CANCEL)}
                  </button>
                </div>
              </form>
            )
          )}
        </>
      )}
      <span role="status" className="sr-only">
        {copyMessage}
      </span>
    </ManifestSubpageLayout>
  );
}
