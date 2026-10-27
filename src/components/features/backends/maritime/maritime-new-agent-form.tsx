import React from "react";
import { useTranslation } from "react-i18next";
import { Loader2 } from "lucide-react";

import {
  buildDefaultAgentName,
  getDefaultMemory,
  getMemoryOptions,
  isValidMaritimeAgentName,
  type MaritimeAccount,
  type NewMaritimeAgentInput,
} from "#/api/maritime/maritime-bridge";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

const NAME_HINT_ID = "maritime-new-agent-name-hint";
const HELPER_CLASS = "-mt-1 text-pretty text-xs text-muted";
const MB_PER_GB = 1024;

interface MaritimeNewAgentFormProps {
  account: MaritimeAccount;
  isCreating: boolean;
  /** What the create is waiting for, shown in place of the hint while busy. */
  progress: string | null;
  error: string | null;
  onSubmit: (input: NewMaritimeAgentInput) => void;
  onCancel: () => void;
}

/** Name and memory for a new OpenHands agent on Maritime. */
export function MaritimeNewAgentForm({
  account,
  isCreating,
  progress,
  error,
  onSubmit,
  onCancel,
}: MaritimeNewAgentFormProps) {
  const { t } = useTranslation("openhands");
  const [name, setName] = React.useState(buildDefaultAgentName);
  const [chosenMemory, setChosenMemory] = React.useState<number | null>(null);
  const trimmedName = name.trim();
  const isNameValid = isValidMaritimeAgentName(trimmedName);
  const showNameError = trimmedName !== "" && !isNameValid;
  const memoryOptions = getMemoryOptions(account);
  const memMb =
    chosenMemory !== null && memoryOptions.includes(chosenMemory)
      ? chosenMemory
      : getDefaultMemory(account);
  const canSubmit = isNameValid && !isCreating;

  const memoryItems = memoryOptions.map((mb) => ({
    key: String(mb),
    label: t(I18nKey.MARITIME$MEMORY_OPTION, {
      gb: Number((mb / MB_PER_GB).toFixed(2)),
    }),
  }));

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!canSubmit) return;
    onSubmit({ name: trimmedName, memMb, origin: window.location.origin });
  };

  return (
    <form
      onSubmit={submit}
      data-testid="maritime-new-agent-form"
      className="flex flex-col gap-4 border-b border-border px-4 pb-4 pt-1"
    >
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="flex min-w-0 flex-col gap-2">
          <SettingsInput
            testId="maritime-new-agent-name"
            name="maritime-new-agent-name"
            type="text"
            label={t(I18nKey.MARITIME$NAME_LABEL)}
            value={name}
            onChange={setName}
            isDisabled={isCreating}
            ariaInvalid={showNameError}
            ariaDescribedBy={NAME_HINT_ID}
            inputClassName={cn(showNameError && "border-danger")}
          />
          <p
            id={NAME_HINT_ID}
            className={cn(HELPER_CLASS, showNameError && "text-feedback-error")}
          >
            {t(I18nKey.MARITIME$NAME_HINT)}
          </p>
        </div>
        <div className="flex min-w-0 flex-col gap-2">
          <SettingsDropdownInput
            testId="maritime-new-agent-memory"
            name="maritime-new-agent-memory"
            label={t(I18nKey.MARITIME$MEMORY_LABEL)}
            items={memoryItems}
            selectedKey={String(memMb)}
            onSelectionChange={(key) =>
              key !== null && setChosenMemory(Number(key))
            }
            onInputChange={() => {}}
            isClearable={false}
            allowsCustomValue={false}
            isDisabled={isCreating}
          />
          <p className={HELPER_CLASS}>{t(I18nKey.MARITIME$MEMORY_HINT)}</p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <p
          role={error ? "alert" : "status"}
          data-testid={error ? "maritime-new-agent-error" : undefined}
          className={cn(
            "min-w-0 flex-1 text-pretty text-xs",
            error ? "text-status-error" : "text-muted",
          )}
        >
          {error ?? progress}
        </p>
        <BrandButton
          type="button"
          variant="secondary"
          onClick={onCancel}
          isDisabled={isCreating}
          testId="maritime-new-agent-cancel"
        >
          {t(I18nKey.BUTTON$CANCEL)}
        </BrandButton>
        <BrandButton
          type="submit"
          variant="primary"
          isDisabled={!canSubmit}
          testId="maritime-new-agent-submit"
          startContent={
            isCreating ? (
              <Loader2 className="size-4 animate-spin" aria-hidden />
            ) : null
          }
        >
          {t(I18nKey.MARITIME$CREATE)}
        </BrandButton>
      </div>
    </form>
  );
}
