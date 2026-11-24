import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import XMarkIcon from "#/icons/x-mark.svg?react";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useAutomationKv } from "#/hooks/query/use-automation-kv";
import { getErrorStatus } from "#/hooks/query/use-settings";
import { useSaveAutomationKv } from "#/hooks/mutation/use-save-automation-kv";
import type { AutomationKvEntry } from "#/types/automation";
import { formatKvValue, resolveKvDrafts } from "#/utils/automation-kv-drafts";
import { displaySuccessToast } from "#/utils/custom-toast-handlers";
import { formatDate } from "#/utils/format-relative-time";
import { modalTitleLgMediumClassName } from "#/utils/modal-classes";
import { cn } from "#/utils/utils";

const EMPTY_ENTRIES: AutomationKvEntry[] = [];
const HTTP_CONFLICT = 409;
const HTTP_PAYLOAD_TOO_LARGE = 413;
const MIN_EDITOR_ROWS = 3;
const MAX_EDITOR_ROWS = 12;

interface KvStoreModalProps {
  automationId: string;
  /** Edit and save values. Without it the dialog is read-only. */
  canEdit: boolean;
  onClose: () => void;
}

interface KvEntryRowProps {
  entry: AutomationKvEntry;
  editable: boolean;
  draft: string | undefined;
  invalid: boolean;
  disabled: boolean;
  onChange: (text: string) => void;
}

function KvEntryRow({
  entry,
  editable,
  draft,
  invalid,
  disabled,
  onChange,
}: KvEntryRowProps) {
  const { t, i18n } = useTranslation("openhands");
  const text = draft ?? formatKvValue(entry.value);

  return (
    <li
      data-testid="automation-kv-entry"
      className="flex flex-col gap-2 rounded-lg border border-border p-3"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="break-all font-mono text-sm font-medium text-contrast">
          {entry.key}
        </span>
        <span className="text-xs text-muted">
          {t(I18nKey.AUTOMATIONS$DETAIL$KV_UPDATED, {
            date: formatDate(entry.updated_at, i18n.language),
          })}
        </span>
      </div>
      {editable ? (
        <>
          <textarea
            aria-label={entry.key}
            aria-invalid={invalid}
            value={text}
            disabled={disabled}
            spellCheck={false}
            rows={Math.min(
              MAX_EDITOR_ROWS,
              Math.max(MIN_EDITOR_ROWS, text.split("\n").length),
            )}
            onChange={(event) => onChange(event.target.value)}
            className={cn(
              "w-full resize-y rounded-md border bg-surface-deep p-2 font-mono text-xs text-content dark:bg-black/30",
              invalid ? "border-field-border-invalid" : "border-border",
            )}
          />
          {invalid ? (
            <p className="text-xs text-danger">
              {t(I18nKey.AUTOMATIONS$DETAIL$KV_INVALID_JSON)}
            </p>
          ) : null}
        </>
      ) : (
        <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-md bg-surface-deep p-2 font-mono text-xs text-content dark:bg-black/30">
          {text}
        </pre>
      )}
    </li>
  );
}

/** Read the store, and for managers edit its values and save them in one write. */
export function KvStoreModal({
  automationId,
  canEdit,
  onClose,
}: KvStoreModalProps) {
  const { t } = useTranslation("openhands");
  const { data, isError, refetch } = useAutomationKv(automationId, true);
  const save = useSaveAutomationKv(automationId);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [confirmingClose, setConfirmingClose] = useState(false);
  const title = t(I18nKey.AUTOMATIONS$DETAIL$KV_STORE);

  const entries = data?.entries ?? EMPTY_ENTRIES;
  const { changes, invalidKeys } = useMemo(
    () => resolveKvDrafts(entries, drafts),
    [entries, drafts],
  );
  const editedCount = changes.length + invalidKeys.size;
  const hasEdits = editedCount > 0;
  const canSave = changes.length > 0 && invalidKeys.size === 0;
  const saveStatus = save.isError ? getErrorStatus(save.error) : undefined;

  const discardEdits = () => {
    setDrafts({});
    setConfirmingClose(false);
    save.reset();
  };

  const handleSave = () => {
    // The lowest version is the oldest read: any run write since then rejects the save.
    const ifVersion = Math.min(...entries.map((entry) => entry.version));
    save.mutate(
      { changes, ifVersion },
      {
        onSuccess: () => {
          setDrafts({});
          displaySuccessToast(t(I18nKey.AUTOMATIONS$DETAIL$KV_SAVED));
        },
      },
    );
  };

  const handleReload = () => {
    discardEdits();
    void refetch();
  };

  const requestClose = () => {
    if (hasEdits) setConfirmingClose(true);
    else onClose();
  };

  const renderBody = () => {
    if (isError) {
      return (
        <p role="alert" className="text-sm text-danger">
          {t(I18nKey.AUTOMATIONS$DETAIL$KV_ERROR)}
        </p>
      );
    }
    if (!data) {
      return (
        <p className="text-sm italic text-muted">
          {t(I18nKey.AUTOMATIONS$DETAIL$KV_LOADING)}
        </p>
      );
    }
    if (entries.length === 0) {
      return (
        <p className="text-sm italic text-muted">
          {t(I18nKey.AUTOMATIONS$DETAIL$KV_EMPTY)}
        </p>
      );
    }
    return (
      <>
        {data.total > entries.length ? (
          <p className="mb-3 text-xs text-muted">
            {t(I18nKey.AUTOMATIONS$DETAIL$KV_TRUNCATED, {
              shown: entries.length,
              total: data.total,
            })}
          </p>
        ) : null}
        <ul className="flex flex-col gap-3">
          {entries.map((entry) => (
            <KvEntryRow
              key={entry.key}
              entry={entry}
              editable={canEdit}
              draft={drafts[entry.key]}
              invalid={invalidKeys.has(entry.key)}
              disabled={save.isPending}
              onChange={(text) =>
                setDrafts((current) => ({ ...current, [entry.key]: text }))
              }
            />
          ))}
        </ul>
      </>
    );
  };

  return (
    <ModalBackdrop
      onClose={requestClose}
      aria-label={title}
      closeOnEscape={!save.isPending}
      closeOnBackdropClick={!save.isPending}
    >
      <div
        data-testid="automation-kv-modal"
        className="flex dialog-content-height w-full max-w-3xl flex-col overflow-y-auto overscroll-contain rounded-xl border border-border bg-surface"
      >
        <div className="sticky top-0 z-10 flex shrink-0 items-center justify-between gap-4 bg-surface px-6 pb-4 pt-6">
          <h2 className={modalTitleLgMediumClassName}>{title}</h2>
          <button
            type="button"
            onClick={requestClose}
            className="text-muted hover:text-foreground"
            aria-label={t(I18nKey.BUTTON$CLOSE)}
          >
            <XMarkIcon className="size-5" />
          </button>
        </div>
        <div className="flex-1 px-6 pb-6">
          {canEdit && entries.length > 0 ? (
            <div
              role="note"
              data-testid="automation-kv-warning"
              className="mb-4 rounded-lg border-l-4 border-feedback-warning bg-warning/10 p-4"
            >
              <p className="mb-1 text-sm font-semibold text-feedback-warning">
                {t(I18nKey.AUTOMATIONS$DETAIL$KV_WARNING_TITLE)}
              </p>
              <p className="text-sm text-contrast">
                {t(I18nKey.AUTOMATIONS$DETAIL$KV_WARNING)}
              </p>
            </div>
          ) : null}
          {renderBody()}
        </div>
        {canEdit && entries.length > 0 ? (
          <div
            data-testid="automation-kv-footer"
            className="sticky bottom-0 z-10 flex shrink-0 flex-col gap-3 border-t border-border bg-surface px-6 py-4"
          >
            {saveStatus === HTTP_CONFLICT ? (
              <div
                role="alert"
                className="flex flex-wrap items-center justify-between gap-3 text-sm text-danger"
              >
                <span>{t(I18nKey.AUTOMATIONS$DETAIL$KV_CONFLICT)}</span>
                <BrandButton
                  testId="automation-kv-reload"
                  type="button"
                  variant="secondary"
                  onClick={handleReload}
                >
                  {t(I18nKey.AUTOMATIONS$DETAIL$KV_RELOAD)}
                </BrandButton>
              </div>
            ) : null}
            {save.isError && saveStatus !== HTTP_CONFLICT ? (
              <p role="alert" className="text-sm text-danger">
                {t(
                  saveStatus === HTTP_PAYLOAD_TOO_LARGE
                    ? I18nKey.AUTOMATIONS$DETAIL$KV_TOO_LARGE
                    : I18nKey.AUTOMATIONS$DETAIL$KV_SAVE_ERROR,
                )}
              </p>
            ) : null}
            <div className="flex flex-wrap items-center justify-between gap-3">
              <span className="text-sm text-muted">
                {confirmingClose
                  ? t(I18nKey.AUTOMATIONS$DETAIL$KV_DISCARD_PROMPT)
                  : hasEdits &&
                    t(I18nKey.AUTOMATIONS$DETAIL$KV_CHANGED, {
                      count: editedCount,
                    })}
              </span>
              {confirmingClose ? (
                <div className="flex gap-2">
                  <BrandButton
                    type="button"
                    variant="secondary"
                    onClick={() => setConfirmingClose(false)}
                  >
                    {t(I18nKey.AUTOMATIONS$DETAIL$KV_KEEP_EDITING)}
                  </BrandButton>
                  <BrandButton
                    testId="automation-kv-confirm-discard"
                    type="button"
                    variant="danger"
                    onClick={onClose}
                  >
                    {t(I18nKey.AUTOMATIONS$DETAIL$KV_DISCARD)}
                  </BrandButton>
                </div>
              ) : (
                <div className="flex gap-2">
                  <BrandButton
                    testId="automation-kv-discard"
                    type="button"
                    variant="secondary"
                    isDisabled={!hasEdits || save.isPending}
                    onClick={discardEdits}
                  >
                    {t(I18nKey.AUTOMATIONS$DETAIL$KV_DISCARD)}
                  </BrandButton>
                  <BrandButton
                    testId="automation-kv-save"
                    type="button"
                    variant="primary"
                    isDisabled={!canSave || save.isPending}
                    onClick={handleSave}
                  >
                    {t(I18nKey.BUTTON$SAVE)}
                  </BrandButton>
                </div>
              )}
            </div>
          </div>
        ) : null}
      </div>
    </ModalBackdrop>
  );
}
