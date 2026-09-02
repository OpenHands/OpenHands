import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { HttpError } from "@openhands/typescript-client";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ApiKeyModalBase } from "#/components/features/settings/api-key-modal-base";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { useMetaProfiles } from "#/hooks/query/use-meta-profiles";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { useProviderConnections } from "#/hooks/query/use-provider-connections";
import { useSaveMetaProfile } from "#/hooks/mutation/use-save-meta-profile";
import { useActivateMetaProfile } from "#/hooks/mutation/use-activate-meta-profile";
import MetaProfilesService, {
  type MetaProfile,
} from "#/api/meta-profiles-service/meta-profiles-service.api";
import ProfilesService, {
  type SaveProfileRequest,
} from "#/api/profiles-service/profiles-service.api";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { MetaProfileEditor } from "./meta-profile-editor";
import { MetaProfileRow } from "./meta-profile-row";
import { DeleteMetaProfileModal } from "./delete-meta-profile-modal";
import {
  DEFAULT_MAX_SCORE_PARETO_META_PROFILE_DEFAULT,
  DEFAULT_MAX_SCORE_PARETO_META_PROFILE_NAME,
  DEFAULT_MIN_COST_PARETO_META_PROFILE_DEFAULT,
  DEFAULT_MIN_COST_PARETO_META_PROFILE_NAME,
} from "./default-meta-profile";
import {
  buildRouterModel,
  collectRequiredRouterModelNames,
} from "./router-profiles";

type ViewMode = "list" | "create" | "edit";
type RouterTemplate = "max-score-pareto" | "min-cost-pareto" | "custom";

interface EditingMetaProfile {
  name: string;
  config: MetaProfile;
}

const CUSTOM_META_PROFILE_CONFIG: MetaProfile = {
  classifier_model: "",
  default_model: "",
  classes: [],
  prompt_template: "",
  model_table: "",
};

export function MetaLlmSettingsView() {
  const { t } = useTranslation("openhands");
  const { data, isLoading, error } = useMetaProfiles();
  const { data: llmProfilesData } = useLlmProfiles();
  const { data: providerConnections } = useProviderConnections();
  const saveMetaProfile = useSaveMetaProfile();
  const activateMetaProfile = useActivateMetaProfile();

  const [view, setView] = useState<ViewMode>("list");
  const [editing, setEditing] = useState<EditingMetaProfile | null>(null);
  const [createInitial, setCreateInitial] = useState<EditingMetaProfile | null>(
    null,
  );
  // Whether the create editor should pre-select a provider connection to
  // populate the router's LLM profiles (true for the built-in Pareto templates,
  // false for a blank custom profile).
  const [createRouterProfilesByDefault, setCreateRouterProfilesByDefault] =
    useState(true);
  const [isTemplateModalOpen, setIsTemplateModalOpen] = useState(false);
  const [nameToDelete, setNameToDelete] = useState<string | null>(null);
  const [isCreatingRouterProfiles, setIsCreatingRouterProfiles] =
    useState(false);
  const [createdProviderConnections, setCreatedProviderConnections] = useState<
    ProviderConnection[]
  >([]);

  const metaProfiles = data?.meta_profiles ?? [];
  const active = data?.active_meta_profile ?? null;
  const availableProfiles = (llmProfilesData?.profiles ?? []).map(
    (p) => p.name,
  );
  const connections = useMemo(() => {
    const byId = new Map<string, ProviderConnection>();
    for (const connection of providerConnections ?? []) {
      byId.set(connection.id, connection);
    }
    for (const connection of createdProviderConnections) {
      byId.set(connection.id, connection);
    }
    return [...byId.values()];
  }, [providerConnections, createdProviderConnections]);
  const existingNames = metaProfiles.map((p) => p.name);
  // A 404 means the backend predates the /api/meta-profiles endpoints
  // (software-agent-sdk #3744). Surface that explicitly instead of a generic
  // error so the page isn't a dead end on older backends.
  const isUnsupportedBackend =
    error instanceof HttpError && error.status === 404;

  const handleActivate = async (name: string) => {
    try {
      await activateMetaProfile.mutateAsync(name);
      displaySuccessToast(t(I18nKey.SETTINGS$META_PROFILE_ACTIVATED, { name }));
    } catch (activateError) {
      const message =
        activateError instanceof Error
          ? activateError.message
          : t(I18nKey.ERROR$GENERIC);
      displayErrorToast(message);
    }
  };

  const handleEdit = async (name: string) => {
    try {
      const detail = await MetaProfilesService.getMetaProfile(name);
      setEditing({ name: detail.name, config: detail.config });
      setView("edit");
    } catch (loadError) {
      const message =
        loadError instanceof Error
          ? loadError.message
          : t(I18nKey.ERROR$GENERIC);
      displayErrorToast(message);
    }
  };

  const handleChooseTemplate = (template: RouterTemplate) => {
    if (template === "max-score-pareto") {
      setCreateInitial({
        name: DEFAULT_MAX_SCORE_PARETO_META_PROFILE_NAME,
        config: DEFAULT_MAX_SCORE_PARETO_META_PROFILE_DEFAULT,
      });
      setCreateRouterProfilesByDefault(true);
    } else if (template === "min-cost-pareto") {
      setCreateInitial({
        name: DEFAULT_MIN_COST_PARETO_META_PROFILE_NAME,
        config: DEFAULT_MIN_COST_PARETO_META_PROFILE_DEFAULT,
      });
      setCreateRouterProfilesByDefault(true);
    } else {
      setCreateInitial({ name: "", config: CUSTOM_META_PROFILE_CONFIG });
      setCreateRouterProfilesByDefault(false);
    }

    setEditing(null);
    setIsTemplateModalOpen(false);
    setView("create");
  };

  // Create an LLM profile for every model the router config needs that does not
  // already exist, linking the chosen provider connection for credentials. The
  // endpoint is derived by convention from the connection's provider
  // (``<provider>/<name>``); the profile name stays the bare model name so the
  // router can match the classifier's answer against it.
  const createMissingRouterLlmProfiles = async (
    config: MetaProfile,
    providerConnectionId: string,
  ) => {
    const connection = connections.find((c) => c.id === providerConnectionId);
    if (!connection) {
      throw new Error("The selected provider connection no longer exists.");
    }

    const existingProfileNames = new Set(
      availableProfiles.map((profileName) => profileName.toLowerCase()),
    );
    const missingNames = collectRequiredRouterModelNames(config).filter(
      (modelName) => !existingProfileNames.has(modelName.toLowerCase()),
    );
    if (missingNames.length === 0) return;

    await Promise.all(
      missingNames.map((modelName) =>
        ProfilesService.saveProfile(modelName, {
          llm: {
            model: buildRouterModel(connection.provider, modelName),
            usage_id: modelName,
            provider_connection_id: connection.id,
          } as SaveProfileRequest["llm"],
          include_secrets: true,
        }),
      ),
    );
  };

  const handleProviderConnectionCreated = (connection: ProviderConnection) => {
    setCreatedProviderConnections((existing) => [
      connection,
      ...existing.filter((item) => item.id !== connection.id),
    ]);
  };

  const handleSave = async (
    name: string,
    config: MetaProfile,
    providerConnectionId: string | null,
  ) => {
    const shouldActivateAfterCreate = view === "create" && active === null;
    try {
      if (view === "create" && providerConnectionId) {
        setIsCreatingRouterProfiles(true);
        await createMissingRouterLlmProfiles(config, providerConnectionId);
      }
      await saveMetaProfile.mutateAsync({ name, config });
      if (shouldActivateAfterCreate) {
        await activateMetaProfile.mutateAsync(name);
      }
      displaySuccessToast(t(I18nKey.SETTINGS$META_PROFILE_SAVED, { name }));
      setView("list");
      setEditing(null);
      setCreateInitial(null);
    } catch (saveError) {
      const message =
        saveError instanceof Error
          ? saveError.message
          : t(I18nKey.ERROR$GENERIC);
      displayErrorToast(message);
    } finally {
      setIsCreatingRouterProfiles(false);
    }
  };

  const handleCancel = () => {
    setView("list");
    setEditing(null);
    setCreateInitial(null);
  };

  if (isUnsupportedBackend) {
    return (
      <p
        data-testid="meta-profile-unsupported"
        className="text-sm text-[var(--oh-muted)]"
      >
        {t(I18nKey.SETTINGS$META_PROFILE_UNSUPPORTED)}
      </p>
    );
  }

  if (view === "create" || view === "edit") {
    return (
      <MetaProfileEditor
        mode={view === "edit" ? "edit" : "create"}
        initialName={view === "edit" ? editing?.name : createInitial?.name}
        initialConfig={
          view === "edit" ? editing?.config : createInitial?.config
        }
        providerConnections={connections}
        selectRouterConnectionByDefault={
          view === "create" ? createRouterProfilesByDefault : false
        }
        availableProfiles={availableProfiles}
        existingNames={existingNames}
        isSaving={saveMetaProfile.isPending || isCreatingRouterProfiles}
        onSave={handleSave}
        onCancel={handleCancel}
        onProviderConnectionCreated={handleProviderConnectionCreated}
      />
    );
  }

  return (
    <>
      <div className="flex flex-col gap-4">
        {availableProfiles.length === 0 ? (
          <p
            data-testid="meta-profile-no-llm-profiles"
            className="text-sm text-[var(--oh-muted)]"
          >
            {t(I18nKey.SETTINGS$META_PROFILE_NO_LLM_PROFILES)}
          </p>
        ) : null}

        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-base font-medium text-white">
            {t(I18nKey.SETTINGS$META_PROFILES_AVAILABLE)}
          </h2>
          <BrandButton
            testId="add-meta-profile"
            type="button"
            variant="secondary"
            className="ml-auto"
            onClick={() => {
              setIsTemplateModalOpen(true);
            }}
          >
            {t(I18nKey.SETTINGS$ADD_META_PROFILE)}
          </BrandButton>
        </div>

        {isLoading ? (
          <div className="flex justify-center py-6">
            <LoadingSpinner size="small" />
          </div>
        ) : null}

        {error ? (
          <p className="text-sm text-red-400">{t(I18nKey.ERROR$GENERIC)}</p>
        ) : null}

        {!isLoading && !error && metaProfiles.length === 0 ? (
          <p
            data-testid="meta-profile-empty"
            className="text-sm text-[var(--oh-muted)]"
          >
            {t(I18nKey.SETTINGS$META_PROFILE_NO_PROFILES)}
          </p>
        ) : null}

        {metaProfiles.length > 0 ? (
          <div className="flex flex-col gap-2" data-testid="meta-profile-list">
            {metaProfiles.map((info) => (
              <MetaProfileRow
                key={info.name}
                info={info}
                isActive={info.name === active}
                onActivate={handleActivate}
                onEdit={handleEdit}
                onDelete={setNameToDelete}
                isActivating={activateMetaProfile.isPending}
              />
            ))}
          </div>
        ) : null}
      </div>

      <DeleteMetaProfileModal
        name={nameToDelete}
        onClose={() => setNameToDelete(null)}
      />
      <ApiKeyModalBase
        isOpen={isTemplateModalOpen}
        title={t(I18nKey.SETTINGS$META_PROFILE_TEMPLATE_TITLE)}
        width="md"
        onClose={() => setIsTemplateModalOpen(false)}
        footer={
          <BrandButton
            testId="meta-profile-template-cancel"
            type="button"
            variant="tertiary"
            onClick={() => setIsTemplateModalOpen(false)}
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
        }
      >
        <div
          data-testid="meta-profile-template-modal"
          className="flex flex-col gap-3"
        >
          <BrandButton
            testId="meta-profile-template-max-score"
            type="button"
            variant="secondary"
            className="justify-start"
            onClick={() => handleChooseTemplate("max-score-pareto")}
          >
            {t(I18nKey.SETTINGS$META_PROFILE_TEMPLATE_MAX_SCORE)}
          </BrandButton>
          <BrandButton
            testId="meta-profile-template-min-cost"
            type="button"
            variant="secondary"
            className="justify-start"
            onClick={() => handleChooseTemplate("min-cost-pareto")}
          >
            {t(I18nKey.SETTINGS$META_PROFILE_TEMPLATE_MIN_COST)}
          </BrandButton>
          <BrandButton
            testId="meta-profile-template-custom"
            type="button"
            variant="secondary"
            className="justify-start"
            onClick={() => handleChooseTemplate("custom")}
          >
            {t(I18nKey.SETTINGS$META_PROFILE_TEMPLATE_CUSTOM)}
          </BrandButton>
        </div>
      </ApiKeyModalBase>
    </>
  );
}
