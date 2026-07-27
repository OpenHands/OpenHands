import { useQueryClient } from "@tanstack/react-query";
import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useCreateSecret } from "#/hooks/mutation/use-create-secret";
import { useUpdateSecret } from "#/hooks/mutation/use-update-secret";
import { SettingsInput } from "../settings-input";
import { cn } from "#/utils/utils";
import {
  formControlMultilineFieldClassName,
  formControlSettingsFieldClassName,
} from "#/utils/form-control-classes";
import { BrandButton } from "../brand-button";
import { useSearchSecrets } from "#/hooks/query/use-get-secrets";
import { OptionalTag } from "../optional-tag";

interface SecretFormProps {
  mode: "add" | "edit";
  selectedSecret: string | null;
  onCancel: () => void;
}

export function SecretForm({
  mode,
  selectedSecret,
  onCancel,
}: SecretFormProps) {
  const queryClient = useQueryClient();
  const { t } = useTranslation("openhands");

  const { data: secrets } = useSearchSecrets();
  const { mutate: createSecret } = useCreateSecret();
  const { mutate: updateSecret } = useUpdateSecret();

  const [error, setError] = React.useState<string | null>(null);

  const secretDescription =
    (mode === "edit" &&
      selectedSecret &&
      secrets
        ?.find((secret) => secret.name === selectedSecret)
        ?.description?.trim()) ||
    "";

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
        onSettled: onCancel,
        onSuccess: invalidateSecrets,
      },
    );
  };

  const handleEditSecret = (
    secretToEdit: string,
    name: string,
    description?: string,
  ) => {
    updateSecret(
      { secretToEdit, name, description },
      {
        onSettled: onCancel,
        onSuccess: invalidateSecrets,
      },
    );
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const formData = new FormData(event.currentTarget);
    const name = formData.get("secret-name")?.toString();
    const value = formData.get("secret-value")?.toString().trim();
    const description = formData.get("secret-description")?.toString();

    if (name) {
      setError(null);

      const isNameAlreadyUsed = secrets?.some(
        (secret) => secret.name === name && secret.name !== selectedSecret,
      );
      if (isNameAlreadyUsed) {
        setError(t(I18nKey.SECRETS$SECRET_ALREADY_EXISTS));
        return;
      }

      if (mode === "add") {
        if (!value) {
          setError(t(I18nKey.SECRETS$SECRET_VALUE_REQUIRED));
          return;
        }

        handleCreateSecret(name, value, description || undefined);
      } else if (mode === "edit" && selectedSecret) {
        handleEditSecret(selectedSecret, name, description || undefined);
      }
    }
  };

  const formTestId = mode === "add" ? "add-secret-form" : "edit-secret-form";

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
        defaultValue={mode === "edit" && selectedSecret ? selectedSecret : ""}
        placeholder={t(I18nKey.SECRETS$API_KEY_EXAMPLE)}
        pattern="^[a-zA-Z][a-zA-Z0-9_]{0,63}$"
        title={t(I18nKey.SETTINGS$SECRET_NAME_PATTERN_TITLE)}
      />
      {error && <p className="text-red-500 text-sm">{error}</p>}

      {mode === "add" && (
        <label className="flex flex-col gap-2.5 w-full min-w-0">
          <span className="text-sm">{t(I18nKey.FORM$VALUE)}</span>
          <textarea
            data-testid="value-input"
            name="secret-value"
            required
            className={cn(
              "resize-none",
              formControlMultilineFieldClassName,
              "placeholder:italic",
              "disabled:bg-[var(--oh-surface-raised)] disabled:border-[var(--oh-border-subtle)] disabled:cursor-not-allowed",
            )}
            rows={8}
          />
        </label>
      )}

      <label className="flex flex-col gap-2.5 w-full min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm">{t(I18nKey.FORM$DESCRIPTION)}</span>
          <OptionalTag />
        </div>
        <input
          data-testid="description-input"
          name="secret-description"
          defaultValue={secretDescription}
          className={cn(
            formControlSettingsFieldClassName,
            "disabled:bg-[var(--oh-surface-raised)] disabled:border-[var(--oh-border-subtle)]",
          )}
        />
      </label>

      <div className="flex items-center gap-4">
        <BrandButton
          testId="cancel-button"
          type="button"
          variant="secondary"
          onClick={onCancel}
        >
          {t(I18nKey.BUTTON$CANCEL)}
        </BrandButton>
        <BrandButton testId="submit-button" type="submit" variant="primary">
          {mode === "add" && t(I18nKey.SECRETS$ADD_SECRET)}
          {mode === "edit" && t(I18nKey.SECRETS$EDIT_SECRET)}
        </BrandButton>
      </div>
    </form>
  );
}
