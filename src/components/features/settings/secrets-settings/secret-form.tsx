import { useQueryClient } from "@tanstack/react-query";
import React from "react";
import { Eye, EyeOff } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useCreateSecret } from "#/hooks/mutation/use-create-secret";
import { useUpdateSecret } from "#/hooks/mutation/use-update-secret";
import { SettingsInput } from "../settings-input";
import { cn } from "#/utils/utils";
import { formControlSettingsFieldClassName } from "#/utils/form-control-classes";
import { BrandButton } from "../brand-button";
import { useSearchSecrets } from "#/hooks/query/use-get-secrets";
import { OptionalTag } from "../optional-tag";
import type { CustomSecretInput } from "#/api/agent-server-adapter";

interface SecretFormProps {
  mode: "add" | "edit";
  selectedSecret: string | null;
  onCancel: () => void;
  onSaved?: (secret: CustomSecretInput) => void;
  /** Name to prefill in add mode. */
  defaultName?: string;
  /** Allow replacing only the requested default name, never another saved name. */
  allowReplaceDefaultName?: boolean;
  /** Keep the form open after a failed save so the user can retry. */
  keepOpenOnError?: boolean;
}

export function SecretForm({
  mode,
  selectedSecret,
  onCancel,
  onSaved,
  defaultName = "",
  keepOpenOnError = false,
  allowReplaceDefaultName = false,
}: SecretFormProps) {
  const queryClient = useQueryClient();
  const { t } = useTranslation("openhands");

  const { data: secrets } = useSearchSecrets();
  const { mutate: createSecret, isPending: isCreating } = useCreateSecret();
  const { mutate: updateSecret, isPending: isUpdating } = useUpdateSecret();
  const isSaving = isCreating || isUpdating;

  const [error, setError] = React.useState<string | null>(null);
  const [name, setName] = React.useState(
    mode === "edit" && selectedSecret ? selectedSecret : defaultName,
  );
  const [value, setValue] = React.useState("");
  const [showValue, setShowValue] = React.useState(false);
  const valueInputId = React.useId();
  const [description, setDescription] = React.useState("");

  const secretDescription =
    (mode === "edit" &&
      selectedSecret &&
      secrets
        ?.find((secret) => secret.name === selectedSecret)
        ?.description?.trim()) ||
    "";
  const initialName = mode === "edit" && selectedSecret ? selectedSecret : "";

  React.useEffect(() => {
    setName(mode === "edit" && selectedSecret ? selectedSecret : defaultName);
    setValue("");
    setShowValue(false);
    setDescription(secretDescription);
    setError(null);
  }, [mode, selectedSecret, secretDescription, defaultName]);

  const closeUnlessKeptOpen = (_data: unknown, error: Error | null) => {
    if (!error || !keepOpenOnError) onCancel();
  };

  const invalidateSecrets = () => {
    queryClient.invalidateQueries({
      queryKey: ["secrets-search"],
    });
    queryClient.invalidateQueries({
      queryKey: ["secrets"],
    });
  };

  const handleCreateSecret = (
    name: string,
    value: string,
    description?: string,
  ) => {
    createSecret(
      { name, value, description },
      {
        onSettled: closeUnlessKeptOpen,
        onSuccess: () => {
          invalidateSecrets();
          onSaved?.({ name, description });
        },
      },
    );
  };

  const handleEditSecret = (
    secretToEdit: string,
    name: string,
    description?: string,
    value?: string,
  ) => {
    updateSecret(
      { secretToEdit, name, description, value },
      {
        onSettled: closeUnlessKeptOpen,
        onSuccess: () => {
          invalidateSecrets();
          onSaved?.({ name, description });
        },
      },
    );
  };

  const isReplacingDefaultName =
    mode === "add" &&
    allowReplaceDefaultName &&
    name.trim() === defaultName &&
    secrets.some((secret) => secret.name === defaultName);

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const trimmedName = name.trim();
    const trimmedValue = value.trim();
    const normalizedDescription = description || undefined;

    if (trimmedName) {
      setError(null);

      const isNameAlreadyUsed = secrets?.some(
        (secret) =>
          secret.name === trimmedName && secret.name !== selectedSecret,
      );
      if (isNameAlreadyUsed && !isReplacingDefaultName) {
        setError(t(I18nKey.SECRETS$SECRET_ALREADY_EXISTS));
        return;
      }

      if (mode === "add") {
        if (!trimmedValue) {
          setError(t(I18nKey.SECRETS$SECRET_VALUE_REQUIRED));
          return;
        }

        handleCreateSecret(trimmedName, trimmedValue, normalizedDescription);
      } else if (mode === "edit" && selectedSecret) {
        // Blank value means "keep the existing secret"; only send a value when
        // the user typed a replacement (#16134).
        handleEditSecret(
          selectedSecret,
          trimmedName,
          normalizedDescription,
          trimmedValue || undefined,
        );
      }
    }
  };

  const formTestId = mode === "add" ? "add-secret-form" : "edit-secret-form";
  const isEditDirty =
    mode === "edit" &&
    (name.trim() !== initialName.trim() ||
      description !== secretDescription ||
      value.trim() !== "");
  const isSubmitDisabled =
    mode === "add"
      ? !name.trim() || !value.trim()
      : mode === "edit"
        ? !isEditDirty
        : false;

  return (
    <form
      data-testid={formTestId}
      onSubmit={handleSubmit}
      className="flex flex-col items-start gap-6"
    >
      <SettingsInput
        testId="name-input"
        name="secret-name"
        type="text"
        label={t(I18nKey.SETTINGS$NAME)}
        className="w-full min-w-0"
        required
        value={name}
        onChange={setName}
        placeholder={t(I18nKey.SECRETS$API_KEY_EXAMPLE)}
        pattern="^[a-zA-Z][a-zA-Z0-9_]{0,63}$"
        title={t(I18nKey.SETTINGS$SECRET_NAME_PATTERN_TITLE)}
      />
      {error && <p className="text-danger text-sm">{error}</p>}

      <div className="flex flex-col gap-2.5 w-full min-w-0">
        <label htmlFor={valueInputId} className="text-sm">
          {mode === "add"
            ? t(I18nKey.FORM$VALUE)
            : t(I18nKey.SECRETS$SECRET_VALUE_LEAVE_BLANK)}
        </label>
        <div className="flex items-center gap-2">
          <input
            id={valueInputId}
            data-testid="value-input"
            name="secret-value"
            type={showValue ? "text" : "password"}
            autoComplete="new-password"
            required={mode === "add"}
            value={value}
            onChange={(event) => setValue(event.currentTarget.value)}
            className={cn(
              formControlSettingsFieldClassName,
              "disabled:bg-surface-raised disabled:border-border-subtle disabled:cursor-not-allowed",
            )}
          />
          <BrandButton
            type="button"
            variant="secondary"
            ariaLabel={t(
              showValue
                ? I18nKey.SECRETS$HIDE_VALUE
                : I18nKey.SECRETS$SHOW_VALUE,
            )}
            isDisabled={isSaving}
            onClick={() => setShowValue((visible) => !visible)}
          >
            {showValue ? (
              <EyeOff size={16} aria-hidden="true" />
            ) : (
              <Eye size={16} aria-hidden="true" />
            )}
          </BrandButton>
        </div>
      </div>

      <label className="flex flex-col gap-2.5 w-full min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm">{t(I18nKey.FORM$DESCRIPTION)}</span>
          <OptionalTag />
        </div>
        <input
          data-testid="description-input"
          name="secret-description"
          value={description}
          onChange={(event) => setDescription(event.currentTarget.value)}
          className={cn(
            formControlSettingsFieldClassName,
            "disabled:bg-surface-raised disabled:border-border-subtle",
          )}
        />
      </label>

      <div className="flex items-center gap-4">
        <BrandButton
          testId="cancel-button"
          type="button"
          variant="secondary"
          onClick={onCancel}
          isDisabled={isSaving}
        >
          {t(I18nKey.BUTTON$CANCEL)}
        </BrandButton>
        <BrandButton
          testId="submit-button"
          type="submit"
          variant="primary"
          isDisabled={isSubmitDisabled || isSaving}
        >
          {mode === "add" &&
            t(
              isReplacingDefaultName
                ? I18nKey.SECRETS$EDIT_SECRET
                : I18nKey.SECRETS$ADD_SECRET,
            )}
          {mode === "edit" && t(I18nKey.SECRETS$EDIT_SECRET)}
        </BrandButton>
      </div>
    </form>
  );
}
