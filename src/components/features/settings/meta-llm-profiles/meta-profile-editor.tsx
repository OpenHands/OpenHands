import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Trash2 } from "lucide-react";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { ProfileNameInput } from "#/components/features/settings/llm-profiles/profile-name-input";
import { Typography } from "#/ui/typography";
import { isProfileNameValid } from "#/utils/derive-profile-name";
import { cn } from "#/utils/utils";
import { formControlMultilineFieldClassName } from "#/utils/form-control-classes";
import { I18nKey } from "#/i18n/declaration";
import type {
  MetaProfile,
  MetaProfileTargetModel,
} from "#/api/meta-profiles-service/meta-profiles-service.api";

interface MetaProfileEditorProps {
  mode: "create" | "edit";
  initialName?: string;
  initialConfig?: MetaProfile;
  /** Names of saved LLM profiles, offered as dropdown options. */
  availableProfiles: string[];
  /**
   * Names of existing meta-profiles. In create mode a name already present
   * here is rejected, so "Add" cannot silently overwrite an existing profile
   * (the backend save contract is create-or-overwrite).
   */
  existingNames?: string[];
  isSaving: boolean;
  onSave: (name: string, config: MetaProfile) => void;
  onCancel: () => void;
}

const EMPTY_CONFIG: MetaProfile = {
  classifier_model: "",
  default_model: "",
  prompt_template: "",
  model_table: "",
  target_models: [],
};

const INSTANCE_TEXT_PLACEHOLDER = /{{\s*instance_text\s*}}/;
const INSTANCE_TEXT_PLACEHOLDER_TEXT = "{{ instance_text }}";
const MODEL_TABLE_PLACEHOLDER_TEXT = "{{ model_table }}";
const PROMPT_TEMPLATE_PLACEHOLDER = `Return JSON with the best model for this task.
${MODEL_TABLE_PLACEHOLDER_TEXT}

Task:
${INSTANCE_TEXT_PLACEHOLDER_TEXT}`;

const normalizeConfig = (config?: MetaProfile): MetaProfile => ({
  classifier_model: config?.classifier_model ?? "",
  default_model: config?.default_model ?? "",
  classes: [],
  prompt_template: config?.prompt_template ?? "",
  model_table: config?.model_table ?? "",
  target_models: config?.target_models ?? [],
});

export function MetaProfileEditor({
  mode,
  initialName = "",
  initialConfig,
  availableProfiles,
  existingNames = [],
  isSaving,
  onSave,
  onCancel,
}: MetaProfileEditorProps) {
  const { t } = useTranslation("openhands");
  const [name, setName] = useState(initialName);
  const [config, setConfig] = useState<MetaProfile>(() =>
    normalizeConfig(initialConfig ?? EMPTY_CONFIG),
  );

  const profileItems = useMemo(
    () => availableProfiles.map((p) => ({ key: p, label: p })),
    [availableProfiles],
  );

  const isEdit = mode === "edit";
  const nameValid = isProfileNameValid(name, { isRequired: true });
  // In create mode, a name that already exists would overwrite that profile
  // (the backend save is create-or-overwrite), so reject it here.
  const isDuplicateName = !isEdit && existingNames.includes(name.trim());
  const targetModelLabels = config.target_models.map((target) =>
    target.model.trim().toLowerCase(),
  );
  const hasDuplicateTargetModelLabels =
    new Set(targetModelLabels).size !== targetModelLabels.length;
  const canSave =
    nameValid &&
    !isDuplicateName &&
    config.classifier_model.trim().length > 0 &&
    config.default_model.trim().length > 0 &&
    INSTANCE_TEXT_PLACEHOLDER.test(config.prompt_template ?? "") &&
    config.target_models.length > 0 &&
    !hasDuplicateTargetModelLabels &&
    config.target_models.every(
      (target) =>
        target.model.trim().length > 0 && target.profile.trim().length > 0,
    );

  const updateTargetModel = (
    index: number,
    patch: Partial<MetaProfileTargetModel>,
  ) => {
    setConfig((prev) => ({
      ...prev,
      target_models: prev.target_models.map((target, i) =>
        i === index ? { ...target, ...patch } : target,
      ),
    }));
  };

  const addTargetModel = () => {
    setConfig((prev) => ({
      ...prev,
      target_models: [...prev.target_models, { model: "", profile: "" }],
    }));
  };

  const removeTargetModel = (index: number) => {
    setConfig((prev) => ({
      ...prev,
      target_models: prev.target_models.filter((_, i) => i !== index),
    }));
  };

  const handleSave = () => {
    if (!canSave || isSaving) return;
    onSave(name.trim(), {
      classifier_model: config.classifier_model.trim(),
      default_model: config.default_model.trim(),
      classes: [],
      prompt_template: (config.prompt_template ?? "").trim(),
      model_table: config.model_table?.trim() || null,
      target_models: config.target_models.map((target) => ({
        model: target.model.trim(),
        profile: target.profile.trim(),
      })),
    });
  };

  return (
    <div className="flex flex-col gap-6" data-testid="meta-profile-editor">
      <Typography.H3>
        {t(
          isEdit
            ? I18nKey.SETTINGS$EDIT_META_PROFILE
            : I18nKey.SETTINGS$NEW_META_PROFILE,
        )}
      </Typography.H3>

      <div className="flex flex-col gap-1">
        <ProfileNameInput
          testId="meta-profile-name-input"
          value={name}
          onChange={setName}
          isDisabled={isEdit || isSaving}
          isRequired
        />
        {isDuplicateName ? (
          <p
            data-testid="meta-profile-name-taken"
            className="text-xs text-red-400"
          >
            {t(I18nKey.SETTINGS$META_PROFILE_NAME_TAKEN)}
          </p>
        ) : null}
      </div>

      <div className="flex flex-col gap-2">
        <SettingsDropdownInput
          testId="meta-profile-classifier-input"
          name="classifier_model"
          label={t(I18nKey.SETTINGS$META_PROFILE_CLASSIFIER)}
          items={profileItems}
          defaultSelectedKey={initialConfig?.classifier_model || undefined}
          allowsCustomValue
          isDisabled={isSaving}
          onInputChange={(value) =>
            setConfig((prev) => ({ ...prev, classifier_model: value }))
          }
          onSelectionChange={(key) =>
            setConfig((prev) => ({
              ...prev,
              classifier_model: key ? String(key) : "",
            }))
          }
        />
        <p className="text-xs text-[var(--oh-muted)]">
          {t(I18nKey.SETTINGS$META_PROFILE_CLASSIFIER_HELP)}
        </p>
      </div>

      <div className="flex flex-col gap-2">
        <SettingsDropdownInput
          testId="meta-profile-default-input"
          name="default_model"
          label={t(I18nKey.SETTINGS$META_PROFILE_DEFAULT)}
          items={profileItems}
          defaultSelectedKey={initialConfig?.default_model || undefined}
          allowsCustomValue
          isDisabled={isSaving}
          onInputChange={(value) =>
            setConfig((prev) => ({ ...prev, default_model: value }))
          }
          onSelectionChange={(key) =>
            setConfig((prev) => ({
              ...prev,
              default_model: key ? String(key) : "",
            }))
          }
        />
        <p className="text-xs text-[var(--oh-muted)]">
          {t(I18nKey.SETTINGS$META_PROFILE_DEFAULT_HELP)}
        </p>
      </div>

      <div className="flex flex-col gap-2">
        <label className="flex flex-col gap-2.5">
          <span className="text-sm">
            {t(I18nKey.SETTINGS$META_PROFILE_PROMPT_TEMPLATE)}
          </span>
          <textarea
            data-testid="meta-profile-prompt-template"
            rows={8}
            spellCheck={false}
            value={config.prompt_template ?? ""}
            placeholder={PROMPT_TEMPLATE_PLACEHOLDER}
            onChange={(event) =>
              setConfig((prev) => ({
                ...prev,
                prompt_template: event.target.value,
              }))
            }
            disabled={isSaving}
            className={cn(
              formControlMultilineFieldClassName,
              "font-mono text-xs",
            )}
          />
        </label>
        <p className="text-xs text-[var(--oh-muted)]">
          {t(I18nKey.SETTINGS$META_PROFILE_PROMPT_TEMPLATE_HELP, {
            instance_text: INSTANCE_TEXT_PLACEHOLDER_TEXT,
            model_table: MODEL_TABLE_PLACEHOLDER_TEXT,
          })}
        </p>
      </div>

      <div className="flex flex-col gap-2">
        <label className="flex flex-col gap-2.5">
          <span className="text-sm">
            {t(I18nKey.SETTINGS$META_PROFILE_MODEL_TABLE)}
          </span>
          <textarea
            data-testid="meta-profile-model-table"
            rows={5}
            spellCheck={false}
            value={config.model_table ?? ""}
            placeholder={t(
              I18nKey.SETTINGS$META_PROFILE_MODEL_TABLE_PLACEHOLDER,
            )}
            onChange={(event) =>
              setConfig((prev) => ({
                ...prev,
                model_table: event.target.value,
              }))
            }
            disabled={isSaving}
            className={cn(
              formControlMultilineFieldClassName,
              "font-mono text-xs",
            )}
          />
        </label>
        <p className="text-xs text-[var(--oh-muted)]">
          {t(I18nKey.SETTINGS$META_PROFILE_MODEL_TABLE_HELP, {
            model_table: MODEL_TABLE_PLACEHOLDER_TEXT,
          })}
        </p>
      </div>

      <div className="flex flex-col gap-3">
        <div className="flex items-center justify-between gap-3">
          <div className="flex flex-col gap-1">
            <h3 className="text-base font-medium text-white">
              {t(I18nKey.SETTINGS$META_PROFILE_TARGET_MODELS)}
            </h3>
            <p className="text-xs text-[var(--oh-muted)]">
              {t(I18nKey.SETTINGS$META_PROFILE_TARGET_MODELS_HELP)}
            </p>
          </div>
          <BrandButton
            testId="meta-profile-add-target-model"
            type="button"
            variant="secondary"
            onClick={addTargetModel}
            isDisabled={isSaving}
          >
            {t(I18nKey.SETTINGS$META_PROFILE_ADD_TARGET_MODEL)}
          </BrandButton>
        </div>

        {config.target_models.length === 0 ? (
          <p
            data-testid="meta-profile-target-models-empty"
            className="text-sm text-[var(--oh-muted)]"
          >
            {t(I18nKey.SETTINGS$META_PROFILE_TARGET_MODELS_EMPTY)}
          </p>
        ) : (
          <ul className="flex flex-col gap-4">
            {config.target_models.map((target, index) => (
              <li
                key={index}
                className="flex flex-col gap-2 sm:flex-row sm:items-end"
              >
                <div className="flex-1">
                  <SettingsInput
                    testId={`meta-profile-target-model-label-${index}`}
                    label={t(I18nKey.SETTINGS$META_PROFILE_TARGET_MODEL_LABEL)}
                    type="text"
                    className="w-full"
                    value={target.model}
                    placeholder={t(
                      I18nKey.SETTINGS$META_PROFILE_TARGET_MODEL_LABEL_PLACEHOLDER,
                    )}
                    onChange={(value) =>
                      updateTargetModel(index, { model: value })
                    }
                    isDisabled={isSaving}
                  />
                </div>
                <div className="flex-1">
                  <SettingsDropdownInput
                    testId={`meta-profile-target-model-profile-${index}`}
                    name={`target_model_profile_${index}`}
                    label={t(
                      I18nKey.SETTINGS$META_PROFILE_TARGET_MODEL_PROFILE,
                    )}
                    items={profileItems}
                    defaultSelectedKey={target.profile || undefined}
                    allowsCustomValue
                    isDisabled={isSaving}
                    onInputChange={(value) =>
                      updateTargetModel(index, { profile: value })
                    }
                    onSelectionChange={(key) =>
                      updateTargetModel(index, {
                        profile: key ? String(key) : "",
                      })
                    }
                  />
                </div>
                <BrandButton
                  testId={`meta-profile-remove-target-model-${index}`}
                  type="button"
                  variant="secondary"
                  onClick={() => removeTargetModel(index)}
                  isDisabled={isSaving}
                  className="shrink-0"
                >
                  <Trash2 size={16} aria-hidden />
                  <span className="sr-only">
                    {t(I18nKey.SETTINGS$META_PROFILE_REMOVE_TARGET_MODEL)}
                  </span>
                </BrandButton>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="flex items-center gap-3">
        <BrandButton
          testId="meta-profile-save"
          type="button"
          variant="primary"
          onClick={handleSave}
          isDisabled={!canSave || isSaving}
        >
          {t(I18nKey.BUTTON$SAVE)}
        </BrandButton>
        <BrandButton
          testId="meta-profile-cancel"
          type="button"
          variant="tertiary"
          onClick={onCancel}
          isDisabled={isSaving}
        >
          {t(I18nKey.BUTTON$CANCEL)}
        </BrandButton>
      </div>
    </div>
  );
}
