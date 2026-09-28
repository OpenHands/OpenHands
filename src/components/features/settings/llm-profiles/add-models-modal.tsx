import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ApiKeyModalBase } from "#/components/features/settings/api-key-modal-base";
import { useProviderModels } from "#/hooks/query/use-provider-models";
import { useSaveLlmProfile } from "#/hooks/mutation/use-save-llm-profile";
import type { SaveProfileRequest } from "#/api/profiles-service/profiles-service.api";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";
import { mapProvider } from "#/utils/map-provider";
import { deriveProfileNameFromModel } from "#/utils/derive-profile-name";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { isSdkHttpStatusError } from "#/api/agent-server-compatibility";
/** Pull the server's own explanation out of an error, when it sent one. */
function getServerDetail(error: unknown): string | null {
  const detail = (error as { response?: { detail?: unknown } })?.response
    ?.detail;
  return typeof detail === "string" && detail.trim() ? detail : null;
}

interface AddModelsModalProps {
  isOpen: boolean;
  existingNames: string[];
  /**
   * Provider connections the user can bulk-add from. Each option in the
   * provider combobox is one of these — picking it drives the model list and
   * links every created profile to that connection's credential.
   */
  connections: ProviderConnection[];
  /**
   * A connection id to preselect when the modal opens (launched from a
   * connection row's "..."). Omit/leave null for the chooser entry point
   * (the top "Add from provider connections" button), where the user picks.
   */
  initialConnectionId?: string | null;
  onClose: () => void;
}

type RowStatus = "idle" | "saving" | "saved" | "failed";

interface ModelRow {
  /** Full model id, e.g. "openhands/deepseek-v4-flash". */
  model: string;
  /** Derived profile name (not user-editable). */
  name: string;
  selected: boolean;
  status: RowStatus;
}

/**
 * Bulk-add models as LLM profiles from a provider connection. One modal serves
 * two entry points: the top "Add from provider connections" button (chooser
 * mode — pick a connection in the combobox) and a connection row's "..."
 * (preselect mode — the combobox opens on that connection). Whichever entry
 * point, the selected connection's provider drives the model list and every
 * created profile is linked to it via `provider_connection_id` — mirroring the
 * link flow in `llm-settings-local-view.tsx` — so the shared credential is
 * wired up out of the box and no key needs to be added afterward.
 */
export function AddModelsModal({
  isOpen,
  existingNames,
  connections,
  initialConnectionId = null,
  onClose,
}: AddModelsModalProps) {
  const { t } = useTranslation("openhands");
  const [selectedConnectionId, setSelectedConnectionId] = useState<
    string | null
  >(null);
  const [rows, setRows] = useState<ModelRow[]>([]);
  const [submitting, setSubmitting] = useState(false);

  const connection = useMemo(
    () => connections.find((c) => c.id === selectedConnectionId) ?? null,
    [connections, selectedConnectionId],
  );
  const provider = connection?.provider ?? null;
  const connectionId = connection?.id ?? null;

  const models = useProviderModels(provider);
  const saveProfile = useSaveLlmProfile();

  // Rows follow the model list, and only the model list. Already-added models
  // are hidden at render rather than rebuilding the rows, because a rebuild
  // discards the selections and per-row statuses the user has accumulated. A
  // row already on screen keeps its state when the list refreshes.
  useEffect(() => {
    const items = models.data ?? [];
    setRows((prev) => {
      const prior = new Map(prev.map((row) => [row.model, row]));
      // Dedupe by canonical model id: a provider's catalog can list the same
      // model twice (e.g. once in the verified set and once in the raw list
      // when the two disagree on prefixing). Without dedup each copy becomes
      // its own row with the same derived name, and the two flag each other as
      // conflicts — surfacing bogus "Name already exists" on a fresh account.
      const seen = new Set<string>();
      const built: ModelRow[] = [];
      for (const m of items) {
        const full =
          m.provider && !m.name.startsWith(`${m.provider}/`)
            ? `${m.provider}/${m.name}`
            : m.name;
        if (seen.has(full)) continue;
        seen.add(full);
        const carried = prior.get(full);
        built.push(
          carried
            ? carried
            : {
                model: full,
                name: deriveProfileNameFromModel(full),
                // Nothing is pre-selected: the server caps how many profiles an
                // account may hold, so defaulting to "all" invites a submission
                // that is mostly refusals. Choosing is the point of the modal.
                selected: false,
                status: "idle" as RowStatus,
              },
        );
      }
      return built;
    });
  }, [models.data]);

  // The manager keeps this component mounted and drives it with `isOpen`, so
  // without an explicit reset a reopened modal still shows the last session's
  // selections and Saved/Failed marks. Re-seed the preselected connection too,
  // so a row's "..." relaunches on its own connection rather than the one the
  // user last picked in a different session.
  useEffect(() => {
    if (isOpen) {
      setSelectedConnectionId(initialConnectionId);
    } else {
      setRows([]);
      setSubmitting(false);
    }
  }, [isOpen, initialConnectionId]);

  const existing = useMemo(() => new Set(existingNames), [existingNames]);

  if (!isOpen) return null;

  // Already-added models are hidden, not disabled: a row whose name matches an
  // existing profile can't be re-created, so showing it (flagged or greyed)
  // just clutters the list. A saved row is exempt — it was added from this
  // very session, so it stays visible with its "Saved" mark.
  const visibleRows = rows.filter(
    (row) => row.status === "saved" || !existing.has(row.name),
  );

  const selectable = visibleRows;
  const selectedRows = selectable.filter((row) => row.selected);

  const setRow = (model: string, patch: Partial<ModelRow>) =>
    setRows((prev) =>
      prev.map((row) => (row.model === model ? { ...row, ...patch } : row)),
    );

  const allSelected =
    selectable.length > 0 && selectedRows.length === selectable.length;

  const toggleAll = () => {
    const next = !allSelected;
    const reachable = new Set(selectable.map((row) => row.model));
    setRows((prev) =>
      prev.map((row) =>
        reachable.has(row.model) ? { ...row, selected: next } : row,
      ),
    );
  };

  const handleSubmit = async () => {
    setSubmitting(true);
    const targets = selectedRows;
    for (const row of targets) setRow(row.model, { status: "saving" });

    // Sequential, not a parallel fan-out: the server enforces a profile limit,
    // and firing every create at once turns one refusal into a wall of
    // identical failures.
    //
    // A 409 is ambiguous. It means the profile ceiling, which every remaining
    // create would hit too — or a name taken since this list loaded, which
    // says nothing about the rows behind it. Stopping on the first one halts
    // a run that would have succeeded; ignoring it spends the whole selection
    // against a wall. So one 409 fails its own row and the run continues, and
    // a second confirms the wall at a cost of exactly one extra request.
    let added = 0;
    let failed = 0;
    let conflicts = 0;
    let blocked = false;
    let blockedReason: string | null = null;

    for (const [i, row] of targets.entries()) {
      try {
        await saveProfile.mutateAsync({
          name: row.name,
          request: {
            // The connection sources the credential, so it replaces any inline
            // api_key/base_url — mirrored on the normal save flow in
            // `llm-settings-local-view.tsx`. `include_secrets: false` because
            // no secret is being sent; the link is by id.
            llm: {
              model: row.model,
              provider_connection_id: connectionId,
            } as SaveProfileRequest["llm"],
            include_secrets: false,
          },
        });
        setRow(row.model, { status: "saved" });
        added += 1;
      } catch (error) {
        // Surface the server's reason — a silent "failed" row is undebuggable.
        console.error(`profile create failed for ${row.model}:`, error);
        setRow(row.model, { status: "failed" });
        failed += 1;
        if (isSdkHttpStatusError(error, 409)) {
          conflicts += 1;
          if (conflicts >= 2) {
            blocked = true;
            // Read the reason off the refusal that actually stopped the run.
            // Carrying an earlier 409's detail forward would report a raced
            // duplicate name as the cause of a halt it had nothing to do with.
            blockedReason = getServerDetail(error);
            // Rows past this one were marked "saving" up front and are now
            // never attempted; leave them idle rather than spinning forever.
            for (const skipped of targets.slice(i + 1)) {
              setRow(skipped.model, { status: "idle" });
            }
            break;
          }
        }
      }
    }

    setSubmitting(false);
    if (blocked) {
      // Quote the server when it explained itself. When it did not, say so
      // plainly: the partial-count wording below counts only the rows that
      // were attempted, so it reads as a smaller failure than it was.
      displayErrorToast(
        blockedReason ??
          t(I18nKey.SETTINGS$MODELS_ADD_BLOCKED, { added: String(added) }),
      );
    } else if (failed === 0) {
      displaySuccessToast(
        t(I18nKey.SETTINGS$MODELS_ADDED, { count: String(added) }),
      );
      onClose();
    } else {
      displayErrorToast(
        t(I18nKey.SETTINGS$MODELS_ADDED_PARTIAL, {
          added: String(added),
          failed: String(failed),
        }),
      );
    }
  };

  const handleClose = () => {
    if (!submitting) onClose();
  };

  const isLoadingModels = connection !== null && models.isLoading;
  // Only flag "empty" once a connection is chosen and its (non-loading) model
  // list came back bare. In chooser mode with nothing picked yet, the
  // combobox is the prompt — there's no list to call empty.
  const showEmpty =
    connection !== null && !isLoadingModels && visibleRows.length === 0;
  const emptyMessage = I18nKey.SETTINGS$ADD_MODELS_EMPTY;

  const footer = (
    <>
      <BrandButton
        type="button"
        variant="tertiary"
        onClick={handleClose}
        isDisabled={submitting}
      >
        {t(I18nKey.BUTTON$CANCEL)}
      </BrandButton>
      <BrandButton
        testId="add-models-submit"
        type="button"
        variant="primary"
        onClick={handleSubmit}
        isDisabled={submitting || selectedRows.length === 0}
      >
        {submitting ? (
          <LoadingSpinner size="small" />
        ) : (
          t(I18nKey.SETTINGS$ADD_N_PROFILES, {
            count: String(selectedRows.length),
          })
        )}
      </BrandButton>
    </>
  );

  return (
    <ApiKeyModalBase
      isOpen
      title={t(I18nKey.SETTINGS$ADD_MODELS_TITLE)}
      footer={footer}
      onClose={handleClose}
    >
      <div data-testid="add-models-modal" className="flex flex-col gap-3">
        <label
          className="flex flex-col gap-2 text-sm text-white"
          data-testid="add-models-connection-summary"
        >
          {t(I18nKey.SETTINGS$ADD_MODELS_PROVIDER_LABEL)}
          <select
            data-testid="add-models-provider"
            className="rounded-md border border-[var(--oh-border)] bg-[var(--oh-background)] px-3 py-2 text-sm text-white"
            value={selectedConnectionId ?? ""}
            onChange={(e) => setSelectedConnectionId(e.target.value || null)}
            disabled={submitting}
          >
            <option value="" disabled>
              {t(I18nKey.SETTINGS$ADD_MODELS_PROVIDER_PLACEHOLDER)}
            </option>
            {connections.map((c) => (
              <option key={c.id} value={c.id}>
                {c.display_name} ({mapProvider(c.provider)})
              </option>
            ))}
          </select>
        </label>

        {connection && (
          <span className="min-w-0 max-w-full truncate text-xs text-[var(--oh-muted)]">
            {t(I18nKey.SETTINGS$ADD_MODELS_CONNECTION_BOUND, {
              provider: connection.provider,
            })}
          </span>
        )}

        {isLoadingModels && (
          <div data-testid="add-models-loading" className="py-4 text-center">
            <LoadingSpinner size="small" />
          </div>
        )}

        {showEmpty && (
          <p
            data-testid="add-models-empty"
            className="text-sm text-[var(--oh-muted)]"
          >
            {t(emptyMessage)}
          </p>
        )}

        {visibleRows.length > 0 && (
          <>
            <label className="flex items-center gap-2 text-sm text-white">
              <input
                data-testid="add-models-select-all"
                type="checkbox"
                checked={allSelected}
                onChange={toggleAll}
                disabled={submitting || selectable.length === 0}
              />
              {t(I18nKey.SETTINGS$ADD_MODELS_SELECT_ALL)}
            </label>
            <ul className="flex max-h-64 flex-col gap-2 overflow-y-auto">
              {visibleRows.map((row) => {
                const disabled = submitting || row.status === "saved";
                // Show the model name without the provider prefix — the
                // prefix is implied by the selected connection and just adds
                // visual noise to every row.
                const shortName = row.model.split("/").pop() ?? row.model;
                return (
                  <li
                    key={row.model}
                    data-testid={`add-models-row-${row.model}`}
                    className="flex items-center gap-2"
                  >
                    <input
                      data-testid={`add-models-check-${row.model}`}
                      type="checkbox"
                      checked={row.selected}
                      onChange={(e) =>
                        setRow(row.model, { selected: e.target.checked })
                      }
                      disabled={disabled}
                    />
                    <span
                      className="min-w-0 flex-1 truncate text-sm text-white"
                      title={row.model}
                    >
                      {shortName}
                    </span>
                    {row.status === "saving" && <LoadingSpinner size="small" />}
                    {row.status === "saved" && (
                      <span className="text-xs text-green-400">
                        {t(I18nKey.SETTINGS$MODEL_ROW_SAVED)}
                      </span>
                    )}
                    {row.status === "failed" && (
                      <span className="text-xs text-red-400">
                        {t(I18nKey.SETTINGS$MODEL_ROW_FAILED)}
                      </span>
                    )}
                  </li>
                );
              })}
            </ul>
          </>
        )}
      </div>
    </ApiKeyModalBase>
  );
}
