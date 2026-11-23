import React from "react";
import { useTranslation } from "react-i18next";
import { Loader2 } from "lucide-react";

import type { TFunction } from "i18next";

import {
  buildNewSessionName,
  isValidAgentName,
  type MarsSandboxSize,
  type NewMarsAgentInput,
} from "#/api/mars/mars-tunnel-backend";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { useMarsSandboxSizes } from "#/hooks/query/use-mars-agents";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

const DEFAULT_NAME_STEM = "openhands";
const NAME_HINT_ID = "digitalocean-new-agent-name-hint";
const HELPER_CLASS = "-mt-1 text-pretty text-xs text-muted";
/** Never a valid slug, so it cannot collide with a real size. */
const DEFAULT_SIZE_KEY = "__default__";
const MB_PER_GB = 1024;

function formatSizeLabel(
  size: MarsSandboxSize,
  t: TFunction,
  language: string,
): string {
  const params = {
    vcpus: size.vcpus,
    memory: Number((size.memory_mb / MB_PER_GB).toFixed(2)),
  };
  if (typeof size.price_per_month_usd !== "number") {
    return t(I18nKey.DO_AGENTS$SANDBOX_SIZE_OPTION, params);
  }
  const price = new Intl.NumberFormat(language, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(size.price_per_month_usd);
  return t(I18nKey.DO_AGENTS$SANDBOX_SIZE_OPTION_PRICE, { ...params, price });
}

interface DigitalOceanNewAgentFormProps {
  isCreating: boolean;
  error: string | null;
  onSubmit: (input: NewMarsAgentInput) => void;
  onCancel: () => void;
}

/** Name and sandbox size for a new OpenHands agent on DigitalOcean. */
export function DigitalOceanNewAgentForm({
  isCreating,
  error,
  onSubmit,
  onCancel,
}: DigitalOceanNewAgentFormProps) {
  const { t, i18n } = useTranslation("openhands");
  const [name, setName] = React.useState(() =>
    buildNewSessionName(DEFAULT_NAME_STEM),
  );
  const [chosenSizeKey, setChosenSizeKey] = React.useState<string | null>(null);
  const { data: sizes } = useMarsSandboxSizes();
  const trimmedName = name.trim();
  const isNameValid = isValidAgentName(trimmedName);
  const showNameError = trimmedName !== "" && !isNameValid;

  const sortedSizes = React.useMemo(
    () =>
      [...(sizes ?? [])].sort(
        (a, b) => a.vcpus - b.vcpus || a.memory_mb - b.memory_mb,
      ),
    [sizes],
  );
  const sizeKey = chosenSizeKey ?? sortedSizes[0]?.slug;
  const canSubmit = isNameValid && sizeKey !== undefined && !isCreating;

  const sizeItems = React.useMemo(
    () => [
      ...sortedSizes.map((size) => ({
        key: size.slug,
        label: formatSizeLabel(size, t, i18n.language),
      })),
      {
        key: DEFAULT_SIZE_KEY,
        label: t(I18nKey.DO_AGENTS$SANDBOX_SIZE_DEFAULT),
      },
    ],
    [sortedSizes, t, i18n.language],
  );

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!canSubmit) return;
    onSubmit({
      name: trimmedName,
      size: sizeKey === DEFAULT_SIZE_KEY ? undefined : sizeKey,
    });
  };

  return (
    <form
      onSubmit={submit}
      data-testid="digitalocean-new-agent-form"
      className="flex flex-col gap-4 border-b border-border px-4 pb-4 pt-1"
    >
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="flex min-w-0 flex-col gap-2">
          <SettingsInput
            testId="digitalocean-new-agent-name"
            name="digitalocean-new-agent-name"
            type="text"
            label={t(I18nKey.DO_AGENTS$NEW_AGENT_NAME_LABEL)}
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
            {t(I18nKey.DO_AGENTS$NEW_AGENT_NAME_HINT)}
          </p>
        </div>
        <div className="flex min-w-0 flex-col gap-2">
          <SettingsDropdownInput
            testId="digitalocean-new-agent-size"
            name="digitalocean-new-agent-size"
            label={t(I18nKey.DO_AGENTS$SANDBOX_SIZE)}
            items={sizeItems}
            selectedKey={sizeKey ?? null}
            onSelectionChange={(key) =>
              key !== null && setChosenSizeKey(String(key))
            }
            onInputChange={() => {}}
            isClearable={false}
            allowsCustomValue={false}
            isDisabled={isCreating}
          />
          <p className={HELPER_CLASS}>
            {t(I18nKey.DO_AGENTS$SANDBOX_SIZE_HINT)}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <p
          role={error ? "alert" : undefined}
          data-testid={error ? "digitalocean-new-agent-error" : undefined}
          className="min-w-0 flex-1 text-pretty text-xs text-status-error"
        >
          {error}
        </p>
        <BrandButton
          type="button"
          variant="secondary"
          onClick={onCancel}
          isDisabled={isCreating}
          testId="digitalocean-new-agent-cancel"
        >
          {t(I18nKey.BUTTON$CANCEL)}
        </BrandButton>
        <BrandButton
          type="submit"
          variant="primary"
          isDisabled={!canSubmit}
          testId="digitalocean-new-agent-submit"
          startContent={
            isCreating ? (
              <Loader2 className="size-4 animate-spin" aria-hidden />
            ) : null
          }
        >
          {isCreating
            ? t(I18nKey.DO_AGENTS$CREATING_AGENT)
            : t(I18nKey.DO_AGENTS$NEW_AGENT_CREATE)}
        </BrandButton>
      </div>
    </form>
  );
}
