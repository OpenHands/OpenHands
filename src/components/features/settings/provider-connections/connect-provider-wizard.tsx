import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { ApiKeyModalBase } from "#/components/features/settings/api-key-modal-base";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { useSearchProviders } from "#/hooks/query/use-search-providers";
import { useProviderModels } from "#/hooks/query/use-provider-models";
import {
  useCreateProviderConnection,
  useCreateProfileFromConnection,
  useDeleteProviderConnection,
  useUpdateProviderConnection,
  useValidateProviderConnection,
} from "#/hooks/mutation/use-provider-connection-mutations";
import type { ValidateConnectionResponse } from "#/api/provider-connections-service";
import type { LLMModel } from "#/api/config-service/config-service.types";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";

interface ConnectProviderWizardProps {
  isOpen: boolean;
  onClose: () => void;
  /** Optional preselected provider (e.g., from an empty-state CTA). */
  defaultProvider?: string;
}

type WizardStep = "connect" | "pick" | "done";

/**
 * Connect-a-Provider wizard implementing the issue #15492 wireframe:
 *   Step 1 — vendor + key (+ helper line)
 *   Step 2 — test connection (on blur or button) → green check / inline error
 *   Step 3 — pick models (checkboxes, Recommended tag, bulk actions, "More
 *            from {vendor}" collapsible for non-recommended models)
 *   Step 4 — save → confirmation summary + toast; creates one profile per
 *            selected model, all sharing the connection's key by reference.
 *
 * The key is stored as a named secret server-side and never returned here.
 * Local-only in this release; the cloud app-server mirror is a follow-up.
 */
export function ConnectProviderWizard({
  isOpen,
  onClose,
  defaultProvider,
}: ConnectProviderWizardProps) {
  const { t } = useTranslation("openhands");
  const { data: providers, isLoading: providersLoading } = useSearchProviders();
  const createConnection = useCreateProviderConnection();
  const updateConnection = useUpdateProviderConnection();
  const deleteConnection = useDeleteProviderConnection();
  const validateExisting = useValidateProviderConnection();
  const createProfileFromConnection = useCreateProfileFromConnection();

  const [provider, setProvider] = useState<string>(defaultProvider ?? "");
  const [key, setKey] = useState("");
  const [label, setLabel] = useState("");
  const [step, setStep] = useState<WizardStep>("connect");
  const [validated, setValidated] = useState<ValidateConnectionResponse | null>(
    null,
  );
  const [submitting, setSubmitting] = useState(false);
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [showMore, setShowMore] = useState(false);
  const [createdCount, setCreatedCount] = useState(0);
  const keyInputRef = useRef<HTMLInputElement>(null);
  // Id of the connection created in this wizard session. Retrying rotates this
  // record's key instead of creating a second one, and closing before a
  // successful save deletes it so a rejected key never leaves an orphan.
  const pendingConnectionId = useRef<string | null>(null);

  // Full provider catalog (with `verified` flags) for the model picker. The
  // validate response models are the catalog too, but useProviderModels gives
  // us the `verified` flag needed for the "Recommended" tag + default selection.
  const { data: catalogModels, isLoading: catalogLoading } = useProviderModels(
    step === "pick" ? provider : null,
  );

  useEffect(() => {
    if (isOpen) {
      setProvider(defaultProvider ?? "");
      setKey("");
      setLabel("");
      setStep("connect");
      setValidated(null);
      setSelectedModels(new Set());
      setShowMore(false);
      setCreatedCount(0);
      pendingConnectionId.current = null;
    }
  }, [isOpen, defaultProvider]);

  const providerItems = useMemo(
    () => (providers ?? []).map((p) => ({ key: p.name, label: p.name })),
    [providers],
  );

  const canTest = Boolean(provider && key.trim()) && !submitting;

  /** Create-or-rotate the pending connection, then validate it (live probe). */
  const runTest = async () => {
    if (!canTest) {
      displayErrorToast(t(I18nKey.SETTINGS$CONNECTION_PROVIDER_REQUIRED));
      return;
    }
    setSubmitting(true);
    try {
      let connectionId = pendingConnectionId.current;
      if (connectionId) {
        await updateConnection.mutateAsync({
          id: connectionId,
          request: {
            key: key.trim(),
            label: label.trim() || undefined,
          },
        });
      } else {
        const conn = await createConnection.mutateAsync({
          provider,
          key: key.trim(),
          label: label.trim() || undefined,
        });
        connectionId = conn.id;
        pendingConnectionId.current = conn.id;
      }
      const result = await validateExisting.mutateAsync(connectionId);
      setValidated(result);
      if (result.ok) {
        // Advance to the model picker. The connection is committed; we only
        // clean it up if the user cancels before selecting any models.
        setStep("pick");
      } else {
        displayErrorToast(
          t(I18nKey.SETTINGS$CONNECTION_INVALID, {
            error: result.error ?? "",
          }),
        );
      }
    } catch (error) {
      displayErrorToast(
        error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
      );
    } finally {
      setSubmitting(false);
    }
  };

  const handleUseDifferentKey = () => {
    setKey("");
    setValidated(null);
    setStep("connect");
    keyInputRef.current?.focus();
  };

  const toggleModel = (name: string) => {
    setSelectedModels((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const verifiedModels = useMemo(
    () => (catalogModels ?? []).filter((m) => m.verified),
    [catalogModels],
  );
  const otherModels = useMemo(
    () => (catalogModels ?? []).filter((m) => !m.verified),
    [catalogModels],
  );

  // Default selection: all recommended models pre-checked, entered once when
  // the catalog first loads in the pick step.
  useEffect(() => {
    if (step === "pick" && catalogModels && selectedModels.size === 0) {
      setSelectedModels(new Set(verifiedModels.map((m) => m.name)));
    }
  }, [step, catalogModels, verifiedModels, selectedModels.size]);

  const selectAllVerified = () =>
    setSelectedModels(new Set(verifiedModels.map((m) => m.name)));
  const selectAll = () =>
    setSelectedModels(new Set((catalogModels ?? []).map((m) => m.name)));
  const clearAll = () => setSelectedModels(new Set());

  const handleSave = async () => {
    const connectionId = pendingConnectionId.current;
    if (!connectionId) return;
    if (selectedModels.size === 0) {
      displayErrorToast(t(I18nKey.SETTINGS$CONNECTION_PICK_REQUIRED));
      return;
    }
    setSubmitting(true);
    try {
      let created = 0;
      // One profile per selected model, each bound to the connection's key by
      // reference (secret:<name>) so a future key rotation updates them all.
      for (const model of selectedModels) {
        await createProfileFromConnection.mutateAsync({
          id: connectionId,
          request: { profileName: model, model },
        });
        created += 1;
      }
      setCreatedCount(created);
      pendingConnectionId.current = null;
      displaySuccessToast(
        t(I18nKey.SETTINGS$CONNECTION_PROFILES_CREATED, {
          count: created,
          provider,
        }),
      );
      setStep("done");
    } catch (error) {
      displayErrorToast(
        error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
      );
    } finally {
      setSubmitting(false);
    }
  };

  // Closing (cancel or dismiss) before a successful save deletes the
  // half-connected record so a rejected key never leaves an orphan.
  const handleClose = () => {
    const orphanId = pendingConnectionId.current;
    if (orphanId) {
      pendingConnectionId.current = null;
      deleteConnection.mutate(orphanId);
    }
    onClose();
  };

  const busy = submitting || validateExisting.isPending;

  // ── Footer varies by step ──────────────────────────────────────────────
  const footer = (
    <>
      <BrandButton
        testId="connect-provider-cancel"
        type="button"
        variant="secondary"
        onClick={handleClose}
        isDisabled={busy}
      >
        {t(I18nKey.BUTTON$CANCEL)}
      </BrandButton>
      {step === "connect" ? (
        <BrandButton
          testId="connect-provider-test"
          type="button"
          variant="primary"
          onClick={runTest}
          isDisabled={!canTest}
          aria-busy={busy}
        >
          {busy
            ? t(I18nKey.STATUS$SAVING)
            : t(I18nKey.SETTINGS$CONNECTION_TEST)}
        </BrandButton>
      ) : null}
      {step === "pick" ? (
        <BrandButton
          testId="connect-provider-save"
          type="button"
          variant="primary"
          onClick={handleSave}
          isDisabled={busy || selectedModels.size === 0}
          aria-busy={busy}
        >
          {busy
            ? t(I18nKey.STATUS$SAVING)
            : t(I18nKey.SETTINGS$CONNECTION_SAVE)}
        </BrandButton>
      ) : null}
      {step === "done" ? (
        <BrandButton
          testId="connect-provider-done"
          type="button"
          variant="primary"
          onClick={handleClose}
        >
          {t(I18nKey.BUTTON$CLOSE)}
        </BrandButton>
      ) : null}
    </>
  );

  return (
    <ApiKeyModalBase
      isOpen={isOpen}
      title={t(I18nKey.SETTINGS$CONNECT_PROVIDER)}
      width="md"
      footer={footer}
      onClose={handleClose}
      initialFocusRef={keyInputRef}
    >
      <div className="flex flex-col gap-4">
        {step === "connect" ? (
          <>
            <SettingsDropdownInput
              testId="connection-provider-select"
              name="connection-provider"
              label={t(I18nKey.SETTINGS$CONNECTION_PROVIDER_LABEL)}
              items={providerItems}
              selectedKey={provider}
              placeholder={t(I18nKey.SETTINGS$CONNECTION_PROVIDER_PLACEHOLDER)}
              isLoading={providersLoading}
              onSelectionChange={(k) => setProvider(k ? String(k) : "")}
            />
            <div className="flex flex-col gap-1">
              <SettingsInput
                testId="connection-api-key"
                name="connection-api-key"
                ref={keyInputRef}
                label={t(I18nKey.SETTINGS$CONNECTION_API_KEY_LABEL)}
                type="password"
                value={key}
                placeholder={t(I18nKey.SETTINGS$CONNECTION_API_KEY_PLACEHOLDER)}
                onChange={setKey}
                showRequiredTag
                onBlur={() => {
                  // Trigger a test when the key field loses focus, if a key is
                  // present and we haven't tested yet (issue wireframe step 2).
                  if (key.trim() && !validated && !busy) runTest();
                }}
              />
              <p className="text-xs leading-4 text-tertiary-light">
                {t(I18nKey.SETTINGS$CONNECTION_KEY_HELPER)}
              </p>
            </div>
            <SettingsInput
              testId="connection-label"
              name="connection-label"
              label={t(I18nKey.SETTINGS$CONNECTION_LABEL_FIELD)}
              type="text"
              value={label}
              onChange={setLabel}
              showOptionalTag
            />
            {busy ? <LoadingSpinner size="small" /> : null}
            {validated && !validated.ok ? (
              <div
                data-testid="connection-invalid-summary"
                className="flex flex-col gap-2 rounded-md border border-danger/40 bg-danger/10 p-3"
              >
                <p className="text-sm leading-5 text-danger">
                  {t(I18nKey.SETTINGS$CONNECTION_INVALID, {
                    error: validated.error ?? "",
                  })}
                </p>
                <div className="flex gap-2">
                  <BrandButton
                    testId="connection-try-again"
                    type="button"
                    variant="secondary"
                    onClick={runTest}
                    isDisabled={busy}
                  >
                    {t(I18nKey.SETTINGS$CONNECTION_TRY_AGAIN)}
                  </BrandButton>
                  <BrandButton
                    testId="connection-different-key"
                    type="button"
                    variant="tertiary"
                    onClick={handleUseDifferentKey}
                    isDisabled={busy}
                  >
                    {t(I18nKey.SETTINGS$CONNECTION_USE_DIFFERENT_KEY)}
                  </BrandButton>
                </div>
              </div>
            ) : null}
          </>
        ) : null}

        {step === "pick" ? (
          <>
            <div
              data-testid="connection-validated-summary"
              className="flex items-center gap-2 rounded-md border border-success/40 bg-success/10 p-3"
            >
              <span className="text-success" aria-hidden>
                {/* eslint-disable-next-line i18next/no-literal-string -- glyph, not translatable */}
                {"✓"}
              </span>
              <p className="text-sm leading-5 text-success">
                {t(I18nKey.SETTINGS$CONNECTION_TEST_SUCCESS, {
                  count: validated?.models.length ?? 0,
                })}
              </p>
            </div>
            <div className="flex flex-col gap-1">
              <h3 className="text-sm font-medium text-white">
                {t(I18nKey.SETTINGS$CONNECTION_PICK_MODELS)}
              </h3>
              <p className="text-xs leading-4 text-tertiary-light">
                {t(I18nKey.SETTINGS$CONNECTION_PICK_MODELS_HINT, { provider })}
              </p>
            </div>
            {catalogLoading ? <LoadingSpinner size="small" /> : null}
            {(catalogModels ?? []).length === 0 && !catalogLoading ? (
              <p
                data-testid="connection-no-catalog"
                className="text-sm leading-5 text-danger"
              >
                {t(I18nKey.SETTINGS$CONNECTION_NO_CATALOG)}
              </p>
            ) : null}
            {/* Bulk actions */}
            <div className="flex flex-wrap gap-2">
              <BrandButton
                testId="connection-select-all-verified"
                type="button"
                variant="tertiary"
                onClick={selectAllVerified}
              >
                {t(I18nKey.SETTINGS$CONNECTION_SELECT_ALL_VERIFIED)}
              </BrandButton>
              <BrandButton
                testId="connection-select-all"
                type="button"
                variant="tertiary"
                onClick={selectAll}
              >
                {t(I18nKey.SETTINGS$CONNECTION_SELECT_ALL)}
              </BrandButton>
              <BrandButton
                testId="connection-clear"
                type="button"
                variant="tertiary"
                onClick={clearAll}
              >
                {t(I18nKey.SETTINGS$CONNECTION_CLEAR)}
              </BrandButton>
            </div>
            {/* Recommended models */}
            <ul
              data-testid="connection-model-list-verified"
              className="flex flex-col gap-1"
            >
              {verifiedModels.map((m) => (
                <ModelRow
                  key={m.name}
                  model={m}
                  checked={selectedModels.has(m.name)}
                  onToggle={toggleModel}
                />
              ))}
            </ul>
            {/* More from {vendor} — non-recommended, collapsible */}
            {otherModels.length > 0 ? (
              <div className="flex flex-col gap-1">
                <button
                  type="button"
                  data-testid="connection-more-toggle"
                  className="text-left text-xs font-medium text-tertiary-light hover:text-white"
                  onClick={() => setShowMore((v) => !v)}
                >
                  {t(I18nKey.SETTINGS$CONNECTION_MORE_FROM, { provider })}
                  {/* eslint-disable-next-line i18next/no-literal-string -- arrow glyphs, not translatable */}
                  {showMore ? " ▾" : " ▸"}
                </button>
                {showMore ? (
                  <ul
                    data-testid="connection-model-list-more"
                    className="flex flex-col gap-1"
                  >
                    {otherModels.map((m) => (
                      <ModelRow
                        key={m.name}
                        model={m}
                        checked={selectedModels.has(m.name)}
                        onToggle={toggleModel}
                      />
                    ))}
                  </ul>
                ) : null}
              </div>
            ) : null}
            {/* Save summary */}
            {selectedModels.size > 0 ? (
              <p
                data-testid="connection-save-summary"
                className="text-sm leading-5 text-tertiary-light"
              >
                {t(I18nKey.SETTINGS$CONNECTION_SAVE_SUMMARY, {
                  count: selectedModels.size,
                  provider,
                })}
              </p>
            ) : null}
          </>
        ) : null}

        {step === "done" ? (
          <p
            data-testid="connection-done-summary"
            className="text-sm leading-5 text-success"
          >
            {t(I18nKey.SETTINGS$CONNECTION_PROFILES_CREATED, {
              count: createdCount,
              provider,
            })}
          </p>
        ) : null}
      </div>
    </ApiKeyModalBase>
  );
}

function ModelRow({
  model,
  checked,
  onToggle,
}: {
  model: LLMModel;
  checked: boolean;
  onToggle: (name: string) => void;
}) {
  const { t } = useTranslation("openhands");
  return (
    <li className="flex items-center gap-2 rounded-md border border-tertiary-light/20 px-2 py-1">
      <label className="flex flex-1 cursor-pointer items-center gap-2">
        <input
          type="checkbox"
          data-testid={`connection-model-${model.name}`}
          checked={checked}
          onChange={() => onToggle(model.name)}
          className="h-4 w-4"
        />
        <span className="text-sm text-white">{model.name}</span>
      </label>
      {model.verified ? (
        <span
          data-testid={`connection-recommended-${model.name}`}
          className="rounded-full bg-success/20 px-2 py-0.5 text-xs text-success"
        >
          {t(I18nKey.SETTINGS$CONNECTION_RECOMMENDED)}
        </span>
      ) : null}
    </li>
  );
}
