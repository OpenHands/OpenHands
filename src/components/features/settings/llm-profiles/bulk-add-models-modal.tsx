import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ApiKeyModalBase } from "#/components/features/settings/api-key-modal-base";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";
import { useSaveLlmProfile } from "#/hooks/mutation/use-save-llm-profile";
import { deriveProfileNameFromModel } from "#/utils/derive-profile-name";
import { getApiErrorMessage } from "#/utils/api-error-message";
import { I18nKey } from "#/i18n/declaration";

interface BulkAddModelsModalProps {
  /** When `null` the modal is closed. */
  connection: ProviderConnection | null;
  /** Names of profiles that already exist, so pasted models that would
   * collide are skipped instead of erroring. */
  existingProfileNames: Set<string>;
  onClose: () => void;
}

/** One parsed model id from the textarea, paired with its derived profile name. */
interface ParsedModel {
  modelId: string;
  profileName: string;
}

type ResultStatus = "created" | "failed";

/**
 * Splits pasted text into model ids: newline- or comma-separated, trimmed,
 * empty entries dropped, duplicates collapsed (first occurrence wins).
 */
function parseModelIds(raw: string): string[] {
  const seen = new Set<string>();
  const ids: string[] = [];
  for (const line of raw.split(/[\n,]/)) {
    const trimmed = line.trim();
    if (!trimmed || seen.has(trimmed)) continue;
    seen.add(trimmed);
    ids.push(trimmed);
  }
  return ids;
}

/**
 * A model id pasted with its own provider prefix (e.g. "openai/gpt-4o") is
 * used verbatim; a bare id (e.g. "gpt-oss:120b") is prefixed with the
 * connection's provider so it routes through the connection correctly.
 */
function buildModelString(connection: ProviderConnection, modelId: string) {
  return modelId.includes("/") ? modelId : `${connection.provider}/${modelId}`;
}

/**
 * Creates one LLM profile per pasted model id, all linked to the same
 * provider connection. There is no live "discover this connection's models"
 * endpoint (that needs a server-side proxy that doesn't exist here — a
 * browser `fetch` to an arbitrary base URL is blocked by CORS), so the user
 * supplies the id list themselves — e.g. from their own `curl .../v1/models`
 * output or the provider's docs.
 */
export function BulkAddModelsModal({
  connection,
  existingProfileNames,
  onClose,
}: BulkAddModelsModalProps) {
  const { t } = useTranslation("openhands");
  const saveProfile = useSaveLlmProfile();
  const [rawInput, setRawInput] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [results, setResults] = useState<
    { modelId: string; status: ResultStatus; error?: string }[] | null
  >(null);

  const { toCreate, toSkip }: { toCreate: ParsedModel[]; toSkip: string[] } =
    useMemo(() => {
      const create: ParsedModel[] = [];
      const skip: string[] = [];
      for (const modelId of parseModelIds(rawInput)) {
        const profileName = deriveProfileNameFromModel(modelId);
        if (existingProfileNames.has(profileName)) {
          skip.push(modelId);
        } else {
          create.push({ modelId, profileName });
        }
      }
      return { toCreate: create, toSkip: skip };
    }, [rawInput, existingProfileNames]);

  if (!connection) return null;

  const handleClose = () => {
    if (isSubmitting) return;
    setRawInput("");
    setResults(null);
    onClose();
  };

  const handleSubmit = async () => {
    if (toCreate.length === 0 || isSubmitting) return;
    setIsSubmitting(true);
    // Sequential, not Promise.all: keeps this from hammering the agent-server
    // with a burst of concurrent saves, and each result is attributable to
    // its own model as it lands rather than resolved all-at-once.
    const outcomes: {
      modelId: string;
      status: ResultStatus;
      error?: string;
    }[] = [];
    for (const { modelId, profileName } of toCreate) {
      try {
        await saveProfile.mutateAsync({
          name: profileName,
          request: {
            llm: {
              model: buildModelString(connection, modelId),
              provider_connection_id: connection.id,
              auth_type: "api_key",
              subscription_vendor: null,
            },
            include_secrets: true,
          },
        });
        outcomes.push({ modelId, status: "created" });
      } catch (error) {
        outcomes.push({
          modelId,
          status: "failed",
          error: getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC)),
        });
      }
    }
    setResults(outcomes);
    setIsSubmitting(false);
  };

  const createdCount =
    results?.filter((r) => r.status === "created").length ?? 0;
  const failed = results?.filter((r) => r.status === "failed") ?? [];

  const footer = results ? (
    <BrandButton
      testId="bulk-add-models-done"
      type="button"
      variant="primary"
      onClick={handleClose}
    >
      {t(I18nKey.BUTTON$CLOSE)}
    </BrandButton>
  ) : (
    <>
      <BrandButton
        type="button"
        variant="tertiary"
        onClick={handleClose}
        isDisabled={isSubmitting}
      >
        {t(I18nKey.BUTTON$CANCEL)}
      </BrandButton>
      <BrandButton
        testId="bulk-add-models-submit"
        type="button"
        variant="primary"
        onClick={handleSubmit}
        isDisabled={isSubmitting || toCreate.length === 0}
        aria-busy={isSubmitting}
      >
        {isSubmitting ? (
          <LoadingSpinner size="small" />
        ) : (
          t(I18nKey.SETTINGS$PROVIDER_CONNECTION_BULK_ADD_SUBMIT, {
            count: toCreate.length,
          })
        )}
      </BrandButton>
    </>
  );

  return (
    <ApiKeyModalBase
      isOpen
      title={t(I18nKey.SETTINGS$PROVIDER_CONNECTION_BULK_ADD_TITLE)}
      footer={footer}
      onClose={handleClose}
    >
      <div data-testid="bulk-add-models-modal" className="flex flex-col gap-4">
        {results ? (
          <div className="flex flex-col gap-2 text-sm">
            <p data-testid="bulk-add-models-result-created">
              {t(I18nKey.SETTINGS$PROVIDER_CONNECTION_BULK_ADD_RESULT_CREATED, {
                count: createdCount,
              })}
            </p>
            {failed.length > 0 ? (
              <p
                data-testid="bulk-add-models-result-failed"
                className="text-danger"
              >
                {t(
                  I18nKey.SETTINGS$PROVIDER_CONNECTION_BULK_ADD_RESULT_FAILED,
                  { names: failed.map((f) => f.modelId).join(", ") },
                )}
              </p>
            ) : null}
          </div>
        ) : (
          <>
            <fieldset className="flex flex-col gap-2.5 w-full">
              <label className="text-sm" htmlFor="bulk-add-models-textarea">
                {t(
                  I18nKey.SETTINGS$PROVIDER_CONNECTION_BULK_ADD_TEXTAREA_LABEL,
                )}
              </label>
              <textarea
                id="bulk-add-models-textarea"
                data-testid="bulk-add-models-textarea"
                className="w-full min-h-32 rounded-lg border border-[var(--oh-border)] bg-base-secondary px-3 py-2 text-sm text-white"
                value={rawInput}
                onChange={(event) => setRawInput(event.target.value)}
                // eslint-disable-next-line i18next/no-literal-string -- example value, not translatable
                placeholder={"gpt-oss:120b-cloud\nglm-5.3-flash:cloud"}
              />
              <p className="text-xs text-[var(--oh-muted)]">
                {t(I18nKey.SETTINGS$PROVIDER_CONNECTION_BULK_ADD_HINT)}
              </p>
            </fieldset>
            {toCreate.length > 0 || toSkip.length > 0 ? (
              <p
                data-testid="bulk-add-models-preview"
                className="text-xs text-[var(--oh-muted)]"
              >
                {t(I18nKey.SETTINGS$PROVIDER_CONNECTION_BULK_ADD_PREVIEW, {
                  create: toCreate.length,
                  skip: toSkip.length,
                })}
              </p>
            ) : null}
          </>
        )}
      </div>
    </ApiKeyModalBase>
  );
}
