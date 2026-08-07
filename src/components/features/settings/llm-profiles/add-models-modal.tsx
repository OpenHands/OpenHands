import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ApiKeyModalBase } from "#/components/features/settings/api-key-modal-base";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { useSearchProviders } from "#/hooks/query/use-search-providers";
import { useProviderModels } from "#/hooks/query/use-provider-models";
import { useSaveLlmProfile } from "#/hooks/mutation/use-save-llm-profile";
import type { SaveProfileRequest } from "#/api/profiles-service/profiles-service.api";
import {
  deriveProfileNameFromModel,
  isProfileNameValid,
} from "#/utils/derive-profile-name";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

interface AddModelsModalProps {
  isOpen: boolean;
  existingNames: string[];
  onClose: () => void;
}

type RowStatus = "idle" | "saving" | "saved" | "failed";

interface ModelRow {
  /** Full model id, e.g. "openhands/deepseek-v4-flash". */
  model: string;
  /** Editable profile name. */
  name: string;
  selected: boolean;
  status: RowStatus;
}

/**
 * Bulk-add a provider's models as LLM profiles: pick a provider, select
 * models, edit the proposed names, add them all at once. Profiles are created
 * keyless (`include_secrets: false`) — a key, when a model needs one, is added
 * by editing the profile afterward, same as any keyless profile.
 */
export function AddModelsModal({
  isOpen,
  existingNames,
  onClose,
}: AddModelsModalProps) {
  const { t } = useTranslation("openhands");
  const [provider, setProvider] = useState<string | null>(null);
  const [verifiedOnly, setVerifiedOnly] = useState(true);
  const [rows, setRows] = useState<ModelRow[]>([]);
  const [submitting, setSubmitting] = useState(false);

  const providers = useSearchProviders();
  const models = useProviderModels(provider);
  const saveProfile = useSaveLlmProfile();

  // Rebuild the rows when the model list or the verified filter changes;
  // names derive once from the model id and stay user-editable afterward.
  useEffect(() => {
    const items = (models.data ?? []).filter(
      (m) => !verifiedOnly || m.verified,
    );
    setRows(
      items.map((m) => {
        const full =
          m.provider && !m.name.startsWith(`${m.provider}/`)
            ? `${m.provider}/${m.name}`
            : m.name;
        return {
          model: full,
          name: deriveProfileNameFromModel(full),
          selected: true,
          status: "idle" as RowStatus,
        };
      }),
    );
  }, [models.data, verifiedOnly]);

  const existing = useMemo(() => new Set(existingNames), [existingNames]);

  if (!isOpen) return null;

  const nameCounts = new Map<string, number>();
  for (const row of rows) {
    nameCounts.set(row.name, (nameCounts.get(row.name) ?? 0) + 1);
  }
  const hasConflict = (row: ModelRow) =>
    existing.has(row.name) || (nameCounts.get(row.name) ?? 0) > 1;

  const isSelectable = (row: ModelRow) =>
    !hasConflict(row) && isProfileNameValid(row.name, { isRequired: true });
  const selectable = rows.filter(isSelectable);
  const selectedRows = selectable.filter((row) => row.selected);

  const setRow = (model: string, patch: Partial<ModelRow>) =>
    setRows((prev) =>
      prev.map((row) => (row.model === model ? { ...row, ...patch } : row)),
    );

  const allSelected =
    selectable.length > 0 && selectedRows.length === selectable.length;

  const toggleAll = () => {
    const next = !allSelected;
    setRows((prev) =>
      prev.map((row) => (isSelectable(row) ? { ...row, selected: next } : row)),
    );
  };

  const handleSubmit = async () => {
    setSubmitting(true);
    const targets = selectedRows;
    for (const row of targets) setRow(row.model, { status: "saving" });

    const results = await Promise.allSettled(
      targets.map((row) =>
        saveProfile.mutateAsync({
          name: row.name,
          request: {
            llm: { model: row.model } as SaveProfileRequest["llm"],
            include_secrets: false,
          },
        }),
      ),
    );

    let failed = 0;
    results.forEach((result, i) => {
      const ok = result.status === "fulfilled";
      if (!ok) {
        failed += 1;
        // Surface the server's reason — a silent "failed" row is undebuggable.
        console.error(
          `profile create failed for ${targets[i].model}:`,
          result.status === "rejected" ? result.reason : result,
        );
      }
      setRow(targets[i].model, { status: ok ? "saved" : "failed" });
    });

    setSubmitting(false);
    const added = targets.length - failed;
    if (failed === 0) {
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

  const providerOptions = providers.data ?? [];
  const isLoadingModels = provider !== null && models.isLoading;
  const showEmpty = provider !== null && !isLoadingModels && rows.length === 0;

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
        <label className="flex flex-col gap-2 text-sm text-white">
          {t(I18nKey.SETTINGS$ADD_MODELS_PROVIDER_LABEL)}
          <select
            data-testid="add-models-provider"
            className="rounded-md border border-[var(--oh-border)] bg-[var(--oh-background)] px-3 py-2 text-sm text-white"
            value={provider ?? ""}
            onChange={(e) => setProvider(e.target.value || null)}
            disabled={submitting}
          >
            <option value="" disabled>
              {t(I18nKey.SETTINGS$ADD_MODELS_PROVIDER_PLACEHOLDER)}
            </option>
            {providerOptions.map((p) => (
              <option key={p.name} value={p.name}>
                {p.name}
              </option>
            ))}
          </select>
        </label>

        {provider !== null && (
          <label className="flex items-center gap-2 text-sm text-white">
            <input
              data-testid="add-models-verified-only"
              type="checkbox"
              checked={verifiedOnly}
              onChange={(e) => setVerifiedOnly(e.target.checked)}
              disabled={submitting}
            />
            {t(I18nKey.SETTINGS$ADD_MODELS_VERIFIED_ONLY)}
          </label>
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
            {t(I18nKey.SETTINGS$ADD_MODELS_EMPTY)}
          </p>
        )}

        {rows.length > 0 && (
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
              {rows.map((row) => {
                // A successfully saved row re-appears in existingNames after
                // the profiles query refreshes — don't flag it as conflicting
                // with itself.
                const conflict = row.status !== "saved" && hasConflict(row);
                const valid = isProfileNameValid(row.name, {
                  isRequired: true,
                });
                const disabled =
                  submitting || conflict || row.status === "saved";
                return (
                  <li
                    key={row.model}
                    data-testid={`add-models-row-${row.model}`}
                    className="flex items-center gap-2"
                  >
                    <input
                      data-testid={`add-models-check-${row.model}`}
                      type="checkbox"
                      checked={row.selected && !conflict}
                      onChange={(e) =>
                        setRow(row.model, { selected: e.target.checked })
                      }
                      disabled={disabled || !valid}
                    />
                    <span
                      className="min-w-0 flex-1 truncate text-sm text-white"
                      title={row.model}
                    >
                      {row.model}
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
                    <div className="w-48">
                      <SettingsInput
                        testId={`add-models-name-${row.model}`}
                        label=""
                        type="text"
                        value={row.name}
                        onChange={(value) => setRow(row.model, { name: value })}
                        isDisabled={disabled}
                        ariaInvalid={!valid || conflict}
                      />
                    </div>
                    {conflict && (
                      <span
                        data-testid={`add-models-conflict-${row.model}`}
                        className={cn("text-xs", "text-red-400")}
                      >
                        {t(I18nKey.SETTINGS$ADD_MODELS_NAME_TAKEN)}
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
