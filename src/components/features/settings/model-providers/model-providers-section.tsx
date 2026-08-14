import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  FaCheck,
  FaCircleCheck,
  FaPencil,
  FaPlus,
  FaTrash,
} from "react-icons/fa6";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ProviderForm } from "./provider-form";
import { PROVIDER_PRESETS, type ProviderPreset } from "./provider-presets";
import { useModelProviders } from "#/hooks/query/use-model-providers";
import {
  useAddProviderModel,
  useDeleteModelProvider,
  useRemoveProviderModel,
  useUpdateProviderModel,
} from "#/hooks/mutation/use-model-provider-mutations";
import type {
  ModelProvider,
  ProviderModel,
} from "#/api/model-providers-service";
import {
  assertProvidersSupportedLocally,
  isModelProvidersNotOnCloudError,
} from "#/api/model-providers-service";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { useCanManageOrgProfiles } from "#/hooks/use-can-manage-org-profiles";

/** GitHub Copilot: a managed provider with no key or configuration. */
function ManagedProviderCard() {
  const { t } = useTranslation("openhands");
  return (
    <li
      data-testid="managed-provider-github-copilot"
      className="flex flex-col gap-3 rounded-md border border-tertiary-light/30 bg-base-secondary p-4"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex flex-col gap-1">
          <span className="text-sm font-medium text-white">
            {t(I18nKey.SETTINGS$GITHUB_COPILOT_PROVIDER)}
          </span>
          <span className="text-xs leading-4 text-tertiary-light">
            {t(I18nKey.SETTINGS$GITHUB_COPILOT_PROVIDER_HINT)}
          </span>
        </div>
        <FaCircleCheck className="mt-0.5 text-success" aria-hidden />
      </div>
      <p className="border-t border-tertiary-light/20 pt-3 text-xs leading-4 text-tertiary-light">
        {t(I18nKey.SETTINGS$GITHUB_COPILOT_PROVIDER_DESCRIPTION)}
      </p>
    </li>
  );
}

/** A single, in-place editable model row nested under a provider. */
function ModelRow({
  provider,
  model,
}: {
  provider: ModelProvider;
  model: ProviderModel;
}) {
  const { t } = useTranslation("openhands");
  const updateModel = useUpdateProviderModel();
  const removeModel = useRemoveProviderModel();
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(model.name);

  const handleSave = async () => {
    if (!name.trim()) {
      displayErrorToast(t(I18nKey.SETTINGS$PROVIDER_MODEL_NAME_REQUIRED));
      return;
    }
    try {
      await updateModel.mutateAsync({
        id: provider.id,
        modelName: model.name,
        model: { name: name.trim(), wireApi: model.wireApi },
      });
      displaySuccessToast(
        t(I18nKey.SETTINGS$PROVIDER_MODEL_UPDATED, { name: name.trim() }),
      );
      setEditing(false);
    } catch (error) {
      displayErrorToast(
        error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
      );
    }
  };

  const handleRemove = async () => {
    try {
      await removeModel.mutateAsync({ id: provider.id, modelName: model.name });
      displaySuccessToast(
        t(I18nKey.SETTINGS$PROVIDER_MODEL_REMOVED, { name: model.name }),
      );
    } catch (error) {
      displayErrorToast(
        error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
      );
    }
  };

  if (editing) {
    return (
      <li className="flex items-end gap-2 rounded-md bg-base-secondary-light/40 px-3 py-2">
        <SettingsInput
          testId={`model-name-input-${model.name}`}
          name={`model-name-${model.name}`}
          label={t(I18nKey.SETTINGS$PROVIDER_MODEL_NAME_LABEL)}
          type="text"
          value={name}
          onChange={setName}
          className="flex-1"
        />
        <BrandButton
          testId={`model-save-${model.name}`}
          type="button"
          variant="primary"
          onClick={handleSave}
          isDisabled={updateModel.isPending}
        >
          {t(I18nKey.BUTTON$SAVE)}
        </BrandButton>
        <BrandButton
          testId={`model-cancel-${model.name}`}
          type="button"
          variant="secondary"
          onClick={() => {
            setName(model.name);
            setEditing(false);
          }}
        >
          {t(I18nKey.BUTTON$CANCEL)}
        </BrandButton>
      </li>
    );
  }

  return (
    <li
      data-testid={`model-row-${model.name}`}
      className="group flex items-center justify-between gap-2 rounded-md bg-base-secondary-light/40 px-3 py-2"
    >
      <span className="text-sm text-white">{model.name}</span>
      <div className="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
        <button
          type="button"
          data-testid={`model-edit-${model.name}`}
          title={t(I18nKey.SETTINGS$PROVIDER_EDIT_MODEL, { name: model.name })}
          aria-label={t(I18nKey.SETTINGS$PROVIDER_EDIT_MODEL, {
            name: model.name,
          })}
          className="rounded p-1 text-tertiary-light hover:text-white"
          onClick={() => setEditing(true)}
        >
          <FaPencil aria-hidden />
        </button>
        <button
          type="button"
          data-testid={`model-remove-${model.name}`}
          title={t(I18nKey.SETTINGS$PROVIDER_DELETE_MODEL, {
            name: model.name,
          })}
          aria-label={t(I18nKey.SETTINGS$PROVIDER_DELETE_MODEL, {
            name: model.name,
          })}
          className="rounded p-1 text-tertiary-light hover:text-danger"
          onClick={handleRemove}
          disabled={removeModel.isPending}
        >
          <FaTrash aria-hidden />
        </button>
      </div>
    </li>
  );
}

/** Inline "add a model" input, revealed by the provider's + button. */
function AddModelRow({
  provider,
  onDone,
}: {
  provider: ModelProvider;
  onDone: () => void;
}) {
  const { t } = useTranslation("openhands");
  const addModel = useAddProviderModel();
  const [name, setName] = useState("");

  const handleAdd = async () => {
    if (!name.trim()) {
      displayErrorToast(t(I18nKey.SETTINGS$PROVIDER_MODEL_NAME_REQUIRED));
      return;
    }
    try {
      await addModel.mutateAsync({
        id: provider.id,
        model: { name: name.trim() },
      });
      displaySuccessToast(
        t(I18nKey.SETTINGS$PROVIDER_MODEL_ADDED, { name: name.trim() }),
      );
      setName("");
      onDone();
    } catch (error) {
      displayErrorToast(
        error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
      );
    }
  };

  return (
    <li className="flex items-end gap-2 rounded-md bg-base-secondary-light/40 px-3 py-2">
      <SettingsInput
        testId={`add-model-input-${provider.id}`}
        name={`add-model-${provider.id}`}
        label={t(I18nKey.SETTINGS$PROVIDER_MODEL_NAME_LABEL)}
        type="text"
        value={name}
        placeholder={t(I18nKey.SETTINGS$PROVIDER_MODEL_NAME_PLACEHOLDER)}
        onChange={setName}
        className="flex-1"
      />
      <BrandButton
        testId={`add-model-confirm-${provider.id}`}
        type="button"
        variant="primary"
        onClick={handleAdd}
        isDisabled={addModel.isPending || !name.trim()}
      >
        {t(I18nKey.SETTINGS$PROVIDER_ADD_MODEL)}
      </BrandButton>
      <BrandButton
        testId={`add-model-cancel-${provider.id}`}
        type="button"
        variant="secondary"
        onClick={onDone}
      >
        {t(I18nKey.BUTTON$CANCEL)}
      </BrandButton>
    </li>
  );
}

/** A configured provider card with its header actions and nested model list. */
function ProviderCard({
  provider,
  onEdit,
}: {
  provider: ModelProvider;
  onEdit: (provider: ModelProvider) => void;
}) {
  const { t } = useTranslation("openhands");
  const deleteProvider = useDeleteModelProvider();
  const [addingModel, setAddingModel] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  const handleDelete = async () => {
    try {
      await deleteProvider.mutateAsync(provider.id);
      displaySuccessToast(
        t(I18nKey.SETTINGS$PROVIDER_DELETED, {
          provider: provider.displayName,
        }),
      );
      setConfirmingDelete(false);
    } catch (error) {
      displayErrorToast(
        error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
      );
    }
  };

  return (
    <li
      data-testid={`provider-card-${provider.id}`}
      className="flex flex-col rounded-md border border-tertiary-light/30 bg-base-secondary"
    >
      <div className="flex items-center justify-between gap-2 p-4">
        <div className="flex flex-col gap-0.5">
          <span className="text-sm font-medium text-white">
            {provider.displayName}
          </span>
          {provider.baseUrl ? (
            <span className="text-xs leading-4 text-tertiary-light">
              {provider.baseUrl}
            </span>
          ) : null}
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            data-testid={`provider-edit-${provider.id}`}
            title={t(I18nKey.SETTINGS$PROVIDER_EDIT, {
              name: provider.displayName,
            })}
            aria-label={t(I18nKey.SETTINGS$PROVIDER_EDIT, {
              name: provider.displayName,
            })}
            className="rounded p-1 text-tertiary-light hover:text-white"
            onClick={() => onEdit(provider)}
          >
            <FaPencil aria-hidden />
          </button>
          {confirmingDelete ? (
            <>
              <button
                type="button"
                data-testid={`provider-delete-confirm-${provider.id}`}
                aria-label={t(I18nKey.BUTTON$CONFIRM)}
                title={t(I18nKey.BUTTON$CONFIRM)}
                className="rounded p-1 text-danger hover:text-danger"
                onClick={handleDelete}
                disabled={deleteProvider.isPending}
              >
                <FaCheck aria-hidden />
              </button>
              <BrandButton
                testId={`provider-delete-cancel-${provider.id}`}
                type="button"
                variant="secondary"
                onClick={() => setConfirmingDelete(false)}
              >
                {t(I18nKey.BUTTON$CANCEL)}
              </BrandButton>
            </>
          ) : (
            <button
              type="button"
              data-testid={`provider-delete-${provider.id}`}
              title={t(I18nKey.SETTINGS$PROVIDER_DELETE, {
                name: provider.displayName,
              })}
              aria-label={t(I18nKey.SETTINGS$PROVIDER_DELETE, {
                name: provider.displayName,
              })}
              className="rounded p-1 text-tertiary-light hover:text-danger"
              onClick={() => setConfirmingDelete(true)}
            >
              <FaTrash aria-hidden />
            </button>
          )}
          {provider.apiKeySet ? (
            <FaCircleCheck
              data-testid={`provider-key-set-${provider.id}`}
              className="ml-1 text-success"
              aria-hidden
            />
          ) : null}
        </div>
      </div>

      <div className="flex flex-col gap-2 border-t border-tertiary-light/20 p-4">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-tertiary-light">
            {t(I18nKey.SETTINGS$PROVIDER_MODELS_COUNT_LABEL, {
              count: provider.models.length,
            })}
          </span>
          <button
            type="button"
            data-testid={`provider-add-model-${provider.id}`}
            title={t(I18nKey.SETTINGS$PROVIDER_ADD_MODEL)}
            aria-label={t(I18nKey.SETTINGS$PROVIDER_ADD_MODEL)}
            className="rounded p-1 text-tertiary-light hover:text-white"
            onClick={() => setAddingModel(true)}
          >
            <FaPlus aria-hidden />
          </button>
        </div>
        <ul className="flex flex-col gap-1">
          {provider.models.map((model) => (
            <ModelRow key={model.name} provider={provider} model={model} />
          ))}
          {addingModel ? (
            <AddModelRow
              provider={provider}
              onDone={() => setAddingModel(false)}
            />
          ) : null}
          {provider.models.length === 0 && !addingModel ? (
            <li className="px-1 py-1 text-xs leading-4 text-tertiary-light">
              {t(I18nKey.SETTINGS$PROVIDER_NO_MODELS)}
            </li>
          ) : null}
        </ul>
      </div>
    </li>
  );
}

/** The "Add provider" button plus its searchable preset dropdown. */
function AddProviderButton({
  onPick,
}: {
  onPick: (preset: ProviderPreset) => void;
}) {
  const { t } = useTranslation("openhands");
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return PROVIDER_PRESETS;
    return PROVIDER_PRESETS.filter(
      (p) =>
        p.label.toLowerCase().includes(q) ||
        p.description.toLowerCase().includes(q),
    );
  }, [search]);

  return (
    <div className="relative">
      <BrandButton
        testId="add-provider-button"
        type="button"
        variant="secondary"
        onClick={() => setOpen((v) => !v)}
      >
        <span className="flex items-center gap-1.5">
          <FaPlus aria-hidden />
          {t(I18nKey.SETTINGS$CONNECT_PROVIDER)}
        </span>
      </BrandButton>
      {open ? (
        <div
          data-testid="add-provider-menu"
          className="absolute right-0 z-20 mt-1 flex w-80 flex-col rounded-md border border-tertiary-light/30 bg-base-secondary p-2 shadow-lg"
        >
          <input
            data-testid="add-provider-search"
            type="text"
            value={search}
            placeholder={t(I18nKey.SETTINGS$PROVIDER_SEARCH_PLACEHOLDER)}
            onChange={(e) => setSearch(e.target.value)}
            className="mb-2 rounded-md border border-tertiary-light/30 bg-base-secondary-light/40 px-3 py-2 text-sm text-white placeholder:text-tertiary-light focus:outline-none"
          />
          <ul className="flex max-h-72 flex-col gap-0.5 overflow-y-auto custom-scrollbar-always">
            {filtered.map((preset) => (
              <li key={preset.kind}>
                <button
                  type="button"
                  data-testid={`add-provider-option-${preset.kind}`}
                  className="flex w-full flex-col gap-0.5 rounded-md px-3 py-2 text-left hover:bg-base-secondary-light/60"
                  onClick={() => {
                    onPick(preset);
                    setOpen(false);
                    setSearch("");
                  }}
                >
                  <span className="text-sm font-medium text-white">
                    {preset.label}
                  </span>
                  <span className="text-xs leading-4 text-tertiary-light">
                    {preset.description}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

/**
 * "Model providers" settings section (issue #15492): connect a provider once,
 * then manage the models under it. Each provider card nests an editable model
 * list with add / edit / remove; the API key is held on the provider and never
 * returned to the client. Local-only in this release; on cloud the section
 * hides itself (the mirror is a follow-up).
 */
export function ModelProvidersSection() {
  const { t } = useTranslation("openhands");
  const canManage = useCanManageOrgProfiles();
  const [formKind, setFormKind] = useState<ProviderPreset | null>(null);
  const [editingProvider, setEditingProvider] = useState<ModelProvider | null>(
    null,
  );

  const { data, isLoading, error } = useModelProviders({ enabled: canManage });

  const cloudUnavailable = (() => {
    try {
      assertProvidersSupportedLocally();
      return false;
    } catch (e) {
      return isModelProvidersNotOnCloudError(e);
    }
  })();

  if (cloudUnavailable) return null;

  const formOpen = Boolean(formKind || editingProvider);
  const closeForm = () => {
    setFormKind(null);
    setEditingProvider(null);
  };

  return (
    <section
      data-testid="model-providers-section"
      className="flex flex-col gap-3"
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="flex flex-col gap-1">
          <h2 className="text-base font-medium text-white">
            {t(I18nKey.SETTINGS$CONFIGURED_PROVIDERS)}
          </h2>
          <p className="max-w-xl text-xs leading-4 text-tertiary-light">
            {t(I18nKey.SETTINGS$PROVIDER_CONNECTIONS_HINT)}
          </p>
        </div>
        {canManage ? <AddProviderButton onPick={setFormKind} /> : null}
      </div>

      {isLoading ? <LoadingSpinner size="small" /> : null}
      {error ? (
        <p
          data-testid="model-providers-error"
          className="text-sm leading-5 text-danger"
        >
          {t(I18nKey.ERROR$GENERIC)}
        </p>
      ) : null}

      <ul className="flex flex-col gap-3">
        <ManagedProviderCard />
        {(data ?? []).map((provider) => (
          <ProviderCard
            key={provider.id}
            provider={provider}
            onEdit={setEditingProvider}
          />
        ))}
      </ul>

      <ProviderForm
        isOpen={formOpen}
        onClose={closeForm}
        kind={formKind?.kind}
        kindLabel={formKind?.label}
        provider={editingProvider ?? undefined}
      />
    </section>
  );
}
