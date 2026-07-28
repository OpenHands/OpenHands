import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Trash2 } from "lucide-react";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { ProfileNameInput } from "#/components/features/settings/llm-profiles/profile-name-input";
import { Typography } from "#/ui/typography";
import { isProfileNameValid } from "#/utils/derive-profile-name";
import { I18nKey } from "#/i18n/declaration";
import type {
  MetaProfile,
  MetaProfileClass,
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
  classes: [],
};

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
  const [config, setConfig] = useState<MetaProfile>(
    initialConfig ?? EMPTY_CONFIG,
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
  const canSave =
    nameValid &&
    !isDuplicateName &&
    config.classifier_model.trim().length > 0 &&
    config.default_model.trim().length > 0 &&
    config.classes.every(
      (c) => c.description.trim().length > 0 && c.model.trim().length > 0,
    );

  const updateClass = (index: number, patch: Partial<MetaProfileClass>) => {
    setConfig((prev) => ({
      ...prev,
      classes: prev.classes.map((c, i) =>
        i === index ? { ...c, ...patch } : c,
      ),
    }));
  };

  const addClass = () => {
    setConfig((prev) => ({
      ...prev,
      classes: [...prev.classes, { description: "", model: "" }],
    }));
  };

  const removeClass = (index: number) => {
    setConfig((prev) => ({
      ...prev,
      classes: prev.classes.filter((_, i) => i !== index),
    }));
  };

  const handleSave = () => {
    if (!canSave || isSaving) return;
    onSave(name.trim(), {
      classifier_model: config.classifier_model.trim(),
      default_model: config.default_model.trim(),
      classes: config.classes.map((c) => ({
        description: c.description.trim(),
        model: c.model.trim(),
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

      <div className="flex flex-col gap-3">
        <div className="flex items-center justify-between gap-3">
          <div className="flex flex-col gap-1">
            <h3 className="text-base font-medium text-white">
              {t(I18nKey.SETTINGS$META_PROFILE_CLASSES)}
            </h3>
            <p className="text-xs text-[var(--oh-muted)]">
              {t(I18nKey.SETTINGS$META_PROFILE_CLASSES_HELP)}
            </p>
          </div>
          <BrandButton
            testId="meta-profile-add-class"
            type="button"
            variant="secondary"
            onClick={addClass}
            isDisabled={isSaving}
          >
            {t(I18nKey.SETTINGS$META_PROFILE_ADD_CLASS)}
          </BrandButton>
        </div>

        {config.classes.length === 0 ? (
          <p
            data-testid="meta-profile-classes-empty"
            className="text-sm text-[var(--oh-muted)]"
          >
            {t(I18nKey.SETTINGS$META_PROFILE_CLASSES_EMPTY)}
          </p>
        ) : (
          <ul className="flex flex-col gap-4">
            {config.classes.map((cls, index) => (
              <li
                key={index}
                className="flex flex-col gap-2 sm:flex-row sm:items-end"
              >
                <div className="flex-1">
                  <SettingsInput
                    testId={`meta-profile-class-description-${index}`}
                    label={t(I18nKey.SETTINGS$META_PROFILE_CLASS_DESCRIPTION)}
                    type="text"
                    className="w-full"
                    value={cls.description}
                    placeholder={t(
                      I18nKey.SETTINGS$META_PROFILE_CLASS_DESCRIPTION_PLACEHOLDER,
                    )}
                    onChange={(value) =>
                      updateClass(index, { description: value })
                    }
                    isDisabled={isSaving}
                  />
                </div>
                <div className="flex-1">
                  <SettingsDropdownInput
                    testId={`meta-profile-class-model-${index}`}
                    name={`class_model_${index}`}
                    label={t(I18nKey.SETTINGS$META_PROFILE_CLASS_MODEL)}
                    items={profileItems}
                    defaultSelectedKey={cls.model || undefined}
                    allowsCustomValue
                    isDisabled={isSaving}
                    onInputChange={(value) =>
                      updateClass(index, { model: value })
                    }
                    onSelectionChange={(key) =>
                      updateClass(index, { model: key ? String(key) : "" })
                    }
                  />
                </div>
                <BrandButton
                  testId={`meta-profile-remove-class-${index}`}
                  type="button"
                  variant="secondary"
                  onClick={() => removeClass(index)}
                  isDisabled={isSaving}
                  className="shrink-0"
                >
                  <Trash2 size={16} aria-hidden />
                  <span className="sr-only">
                    {t(I18nKey.SETTINGS$META_PROFILE_REMOVE_CLASS)}
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
