import {
  Autocomplete,
  AutocompleteItem,
  AutocompleteSection,
} from "@heroui/react";
import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { mapProvider } from "#/utils/map-provider";
import { extractModelAndProvider } from "#/utils/extract-model-and-provider";
import { cn } from "#/utils/utils";
import { formControlSettingsFieldClassName } from "#/utils/form-control-classes";
import { heroUiAutocompleteSelectorButtonClassName } from "#/ui/combobox-caret";
import { HelpLink } from "#/ui/help-link";
import { PRODUCT_URL } from "#/utils/constants";
import { useSearchProviders } from "#/hooks/query/use-search-providers";
import { useProviderModels } from "#/hooks/query/use-provider-models";
import { FREE_MODEL_BADGE_LABEL } from "#/utils/format-model-name";
import { FreeOpenHandsModelsNote } from "#/components/shared/free-models-note";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { Typography } from "#/ui/typography";

/**
 * Dropdown sentinel for "the model I want is not listed". The provider's
 * catalog is whatever LiteLLM knows about, so aggregators such as OpenRouter
 * always lag behind the models they actually serve.
 */
const CUSTOM_MODEL_KEY = "__custom_model__";

const freeModelBadgeClassName =
  "shrink-0 rounded-full border border-warning/40 bg-warning/10 px-1.5 py-0.5 text-[10px] leading-none text-warning";

interface ModelSelectorProps {
  isDisabled?: boolean;
  currentModel?: string;
  onChange?: (provider: string | null, model: string | null) => void;
  onDefaultValuesChanged?: (
    provider: string | null,
    model: string | null,
  ) => void;
  wrapperClassName?: string;
  labelClassName?: string;
}

export function ModelSelector({
  isDisabled,
  currentModel,
  onChange,
  onDefaultValuesChanged,
  wrapperClassName,
  labelClassName,
}: ModelSelectorProps) {
  const [, setLitellmId] = React.useState<string | null>(null);
  const [selectedProvider, setSelectedProvider] = React.useState<string | null>(
    null,
  );
  const [selectedModel, setSelectedModel] = React.useState<string | null>(null);
  const [isCustomModel, setIsCustomModel] = React.useState(false);

  const { data: providers = [] } = useSearchProviders();
  const {
    data: providerModels = [],
    isLoading: isLoadingModels,
    error: modelsError,
  } = useProviderModels(selectedProvider);

  const verifiedProviders = React.useMemo(
    () => providers.filter((p) => p.verified),
    [providers],
  );
  const unverifiedProviders = React.useMemo(
    () => providers.filter((p) => !p.verified),
    [providers],
  );

  const verifiedModels = React.useMemo(
    () => providerModels.filter((m) => m.verified),
    [providerModels],
  );
  const unverifiedModels = React.useMemo(
    () => providerModels.filter((m) => !m.verified),
    [providerModels],
  );

  // DB-driven set of free model names for the selected provider. Mirrors the
  // `verified` flag: the frontend no longer hardcodes which models are free.
  const freeModelNames = React.useMemo(
    () => providerModels.filter((m) => m.free).map((m) => m.name),
    [providerModels],
  );
  const freeModelNameSet = React.useMemo(
    () => new Set(freeModelNames),
    [freeModelNames],
  );

  React.useEffect(() => {
    if (currentModel) {
      const { provider, model } = extractModelAndProvider(currentModel);

      setLitellmId(currentModel);
      setSelectedProvider(provider || null);
      setSelectedModel(model);
      onDefaultValuesChanged?.(provider || null, model);
    }
  }, [currentModel]);

  const handleChangeProvider = (provider: string) => {
    setSelectedProvider(provider);
    setSelectedModel(null);
    setIsCustomModel(false);
    setLitellmId(`${provider}/`);
    onChange?.(provider, null);
  };

  const handleChangeModel = (model: string) => {
    // Switching to free text keeps the current model as the starting point, so
    // the form never holds a model the user can no longer see. With no model to
    // carry over, the empty string clears the form's value rather than leaving
    // the previously selected model behind an empty-looking field.
    if (model === CUSTOM_MODEL_KEY) {
      setIsCustomModel(true);
      onChange?.(selectedProvider, selectedModel ?? "");
      return;
    }

    let fullModel = `${selectedProvider}/${model}`;
    if (selectedProvider === "openai") {
      fullModel = model;
    }
    setIsCustomModel(false);
    setLitellmId(fullModel);
    setSelectedModel(model);
    onChange?.(selectedProvider, model);
  };

  const handleChangeCustomModel = (model: string) => {
    setSelectedModel(model || null);
    setLitellmId(model ? `${selectedProvider}/${model}` : null);
    // Report the empty string rather than null: null means "no model chosen
    // yet" (a provider change), while an emptied custom field must clear the
    // model so the caller's required-field check sees it.
    onChange?.(selectedProvider, model);
  };

  const clear = () => {
    setSelectedProvider(null);
    setLitellmId(null);
  };

  // A model the provider's catalog does not list is edited as free text. The
  // catalog is only authoritative once it has loaded, so an empty list is
  // treated as "not known yet" rather than as "nothing matches".
  const isListedModel = providerModels.some(
    (model) => model.name === selectedModel,
  );
  const showCustomModelInput =
    isCustomModel ||
    (!!selectedModel && providerModels.length > 0 && !isListedModel);
  const modelSelectorKey = showCustomModelInput
    ? CUSTOM_MODEL_KEY
    : selectedModel;

  const isSelectedModelFree = Boolean(
    !showCustomModelInput &&
    selectedModel &&
    freeModelNameSet.has(selectedModel),
  );
  const selectedModelMeasureRef = React.useRef<HTMLSpanElement>(null);
  const [selectedModelTextWidth, setSelectedModelTextWidth] = React.useState(0);

  React.useLayoutEffect(() => {
    if (!isSelectedModelFree || !selectedModelMeasureRef.current) {
      setSelectedModelTextWidth(0);
      return undefined;
    }

    const measureSelectedModel = () => {
      setSelectedModelTextWidth(
        Math.ceil(
          selectedModelMeasureRef.current?.getBoundingClientRect().width ?? 0,
        ),
      );
    };
    measureSelectedModel();

    if (typeof ResizeObserver === "undefined") {
      return undefined;
    }

    const observer = new ResizeObserver(measureSelectedModel);
    observer.observe(selectedModelMeasureRef.current);
    return () => observer.disconnect();
  }, [isSelectedModelFree, selectedModel]);

  const { t } = useTranslation("openhands");

  return (
    <div
      className={cn(
        "flex flex-col md:flex-row w-full min-w-0 justify-between gap-4 md:gap-[46px]",
        wrapperClassName,
      )}
    >
      <fieldset className="flex flex-col gap-2.5 w-full">
        <label className={cn("text-sm", labelClassName)}>
          {t(I18nKey.LLM$PROVIDER)}
        </label>
        <Autocomplete
          data-testid="llm-provider-input"
          isRequired
          isVirtualized={false}
          name="llm-provider-input"
          isDisabled={isDisabled}
          aria-label={t(I18nKey.LLM$PROVIDER)}
          isClearable={false}
          onSelectionChange={(e) => {
            if (e?.toString()) handleChangeProvider(e.toString());
          }}
          onInputChange={(value) => !value && clear()}
          defaultSelectedKey={selectedProvider ?? undefined}
          selectedKey={selectedProvider}
          classNames={{
            popoverContent:
              "bg-content1 rounded-xl border border-[var(--oh-border)]",
            selectorButton: heroUiAutocompleteSelectorButtonClassName,
          }}
          selectorButtonProps={{ disableRipple: true }}
          inputProps={{
            classNames: {
              inputWrapper: formControlSettingsFieldClassName,
            },
          }}
        >
          <AutocompleteSection
            title={t(I18nKey.MODEL_SELECTOR$VERIFIED)}
            classNames={{ heading: "text-[var(--oh-muted)]" }}
          >
            {verifiedProviders.map((provider) => (
              <AutocompleteItem
                data-testid={`provider-item-${provider.name}`}
                key={provider.name}
              >
                {mapProvider(provider.name)}
              </AutocompleteItem>
            ))}
          </AutocompleteSection>
          {unverifiedProviders.length > 0 ? (
            <AutocompleteSection
              title={t(I18nKey.MODEL_SELECTOR$OTHERS)}
              classNames={{ heading: "text-[var(--oh-muted)]" }}
            >
              {unverifiedProviders.map((provider) => (
                <AutocompleteItem key={provider.name}>
                  {mapProvider(provider.name)}
                </AutocompleteItem>
              ))}
            </AutocompleteSection>
          ) : null}
        </Autocomplete>
      </fieldset>

      {selectedProvider === "openhands" && (
        <div className="flex flex-col gap-2">
          <HelpLink
            testId="openhands-account-help"
            text={t(I18nKey.SETTINGS$NEED_OPENHANDS_ACCOUNT)}
            linkText={t(I18nKey.SETTINGS$CLICK_HERE)}
            href={PRODUCT_URL.PRODUCTION}
            size="settings"
            linkColor="white"
          />
        </div>
      )}

      <fieldset className="flex flex-col gap-2.5 w-full">
        <label className={cn("text-sm", labelClassName)}>
          {t(I18nKey.LLM$MODEL)}
        </label>
        <div className="relative">
          <Autocomplete
            data-testid="llm-model-input"
            isRequired
            isVirtualized={false}
            isLoading={isLoadingModels}
            // `SettingsForm` reads the model straight off the form, so the
            // field that actually holds it owns the name: in custom mode this
            // combobox only holds the "Custom Model" label.
            name={showCustomModelInput ? undefined : "llm-model-input"}
            aria-label={t(I18nKey.LLM$MODEL)}
            isClearable={false}
            onSelectionChange={(e) => {
              if (e?.toString()) handleChangeModel(e.toString());
            }}
            isDisabled={isDisabled || !selectedProvider}
            selectedKey={modelSelectorKey}
            defaultSelectedKey={modelSelectorKey ?? undefined}
            classNames={{
              popoverContent:
                "bg-content1 rounded-xl border border-[var(--oh-border)]",
              selectorButton: heroUiAutocompleteSelectorButtonClassName,
            }}
            selectorButtonProps={{ disableRipple: true }}
            inputProps={{
              classNames: {
                inputWrapper: formControlSettingsFieldClassName,
              },
            }}
          >
            <AutocompleteSection
              title={t(I18nKey.MODEL_SELECTOR$VERIFIED)}
              classNames={{ heading: "text-[var(--oh-muted)]" }}
            >
              {verifiedModels.map((model) => (
                <AutocompleteItem key={model.name} textValue={model.name}>
                  <span className="flex min-w-0 items-center gap-2">
                    <span className="truncate">{model.name}</span>
                    {model.free ? (
                      <span className={freeModelBadgeClassName}>
                        {FREE_MODEL_BADGE_LABEL}
                      </span>
                    ) : null}
                  </span>
                </AutocompleteItem>
              ))}
            </AutocompleteSection>
            {unverifiedModels.length > 0 ? (
              <AutocompleteSection
                title={t(I18nKey.MODEL_SELECTOR$OTHERS)}
                classNames={{ heading: "text-[var(--oh-muted)]" }}
              >
                {unverifiedModels.map((model) => (
                  <AutocompleteItem
                    data-testid={`model-item-${model.name}`}
                    key={model.name}
                    textValue={model.name}
                  >
                    {model.name}
                  </AutocompleteItem>
                ))}
              </AutocompleteSection>
            ) : null}
            <AutocompleteItem
              data-testid="model-item-custom"
              key={CUSTOM_MODEL_KEY}
              textValue={t(I18nKey.SETTINGS$CUSTOM_MODEL)}
            >
              {t(I18nKey.SETTINGS$CUSTOM_MODEL)}
            </AutocompleteItem>
          </Autocomplete>
          {isSelectedModelFree && selectedModel ? (
            <>
              <span
                ref={selectedModelMeasureRef}
                className="pointer-events-none absolute left-3 top-1/2 whitespace-pre text-sm opacity-0"
                aria-hidden
              >
                {selectedModel}
              </span>
              <span
                data-testid="selected-free-model-badge"
                className={cn(
                  freeModelBadgeClassName,
                  "pointer-events-none absolute top-1/2 z-10 -translate-y-1/2",
                )}
                style={{
                  left: `calc(0.75rem + ${selectedModelTextWidth}px + 0.5rem)`,
                }}
              >
                {FREE_MODEL_BADGE_LABEL}
              </span>
            </>
          ) : null}
        </div>
        {showCustomModelInput ? (
          <div className="flex flex-col gap-1.5">
            <SettingsInput
              testId="custom-model-input"
              name="llm-model-input"
              label={t(I18nKey.SETTINGS$CUSTOM_MODEL)}
              type="text"
              className="w-full"
              value={selectedModel ?? ""}
              onChange={handleChangeCustomModel}
              isDisabled={isDisabled}
            />
            <Typography.Text className="text-xs text-[var(--oh-muted)]">
              {t(I18nKey.MODEL_SELECTOR$CUSTOM_MODEL_HINT)}
            </Typography.Text>
          </div>
        ) : null}
        {modelsError && (
          <p data-testid="models-error" className="text-danger text-xs">
            {t(I18nKey.CONFIGURATION$ERROR_FETCH_MODELS)}
          </p>
        )}
        {selectedProvider === "openhands" && freeModelNames.length > 0 ? (
          <FreeOpenHandsModelsNote
            modelIds={freeModelNames.map(
              (name) => `${selectedProvider}/${name}`,
            )}
          />
        ) : null}
      </fieldset>
    </div>
  );
}
