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
import { HelpLink } from "#/ui/help-link";
import { PRODUCT_URL } from "#/utils/constants";
import { useSearchProviders } from "#/hooks/query/use-search-providers";
import { useProviderModels } from "#/hooks/query/use-provider-models";

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
    setLitellmId(`${provider}/`);
    onChange?.(provider, null);
  };

  const handleChangeModel = (model: string) => {
    let fullModel = `${selectedProvider}/${model}`;
    if (selectedProvider === "openai") {
      // LiteLLM lists OpenAI models without the openai/ prefix
      fullModel = model;
    }
    setLitellmId(fullModel);
    setSelectedModel(model);
    onChange?.(selectedProvider, model);
  };

  const clear = () => {
    setSelectedProvider(null);
    setLitellmId(null);
  };

  const { t } = useTranslation();

  return (
    <div
      className={cn(
        "flex flex-col md:flex-row w-full max-w-[680px] justify-between gap-4 md:gap-[46px]",
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
          placeholder={t(I18nKey.LLM$SELECT_PROVIDER_PLACEHOLDER)}
          isClearable={false}
          onSelectionChange={(e) => {
            if (e?.toString()) handleChangeProvider(e.toString());
          }}
          onInputChange={(value) => !value && clear()}
          defaultSelectedKey={selectedProvider ?? undefined}
          selectedKey={selectedProvider}
          classNames={{
            popoverContent: "bg-tertiary rounded-xl border border-[#717888]",
          }}
          inputProps={{
            classNames: {
              inputWrapper:
                "bg-tertiary border border-[#717888] h-10 w-full rounded-sm p-2 placeholder:italic",
            },
          }}
        >
          <AutocompleteSection title={t(I18nKey.MODEL_SELECTOR$VERIFIED)}>
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
            <AutocompleteSection title={t(I18nKey.MODEL_SELECTOR$OTHERS)}>
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
        <HelpLink
          testId="openhands-account-help"
          text={t(I18nKey.SETTINGS$NEED_OPENHANDS_ACCOUNT)}
          linkText={t(I18nKey.SETTINGS$CLICK_HERE)}
          href={PRODUCT_URL.PRODUCTION}
          size="settings"
          linkColor="white"
        />
      )}

      <fieldset className="flex flex-col gap-2.5 w-full">
        <label className={cn("text-sm", labelClassName)}>
          {t(I18nKey.LLM$MODEL)}
        </label>
        <Autocomplete
          data-testid="llm-model-input"
          isRequired
          isVirtualized={false}
          isLoading={isLoadingModels}
          name="llm-model-input"
          aria-label={t(I18nKey.LLM$MODEL)}
          placeholder={t(I18nKey.LLM$SELECT_MODEL_PLACEHOLDER)}
          isClearable={false}
          onSelectionChange={(e) => {
            if (e?.toString()) handleChangeModel(e.toString());
          }}
          isDisabled={isDisabled || !selectedProvider}
          selectedKey={selectedModel}
          defaultSelectedKey={selectedModel ?? undefined}
          classNames={{
            popoverContent: "bg-tertiary rounded-xl border border-[#717888]",
          }}
          inputProps={{
            classNames: {
              inputWrapper:
                "bg-tertiary border border-[#717888] h-10 w-full rounded-sm p-2 placeholder:italic",
            },
          }}
        >
          <AutocompleteSection title={t(I18nKey.MODEL_SELECTOR$VERIFIED)}>
            {verifiedModels.map((model) => (
              <AutocompleteItem key={model.name}>{model.name}</AutocompleteItem>
            ))}
          </AutocompleteSection>
          {unverifiedModels.length > 0 ? (
            <AutocompleteSection title={t(I18nKey.MODEL_SELECTOR$OTHERS)}>
              {unverifiedModels.map((model) => (
                <AutocompleteItem
                  data-testid={`model-item-${model.name}`}
                  key={model.name}
                >
                  {model.name}
                </AutocompleteItem>
              ))}
            </AutocompleteSection>
          ) : null}
        </Autocomplete>
        {modelsError && (
          <p data-testid="models-error" className="text-danger text-xs">
            {t(I18nKey.CONFIGURATION$ERROR_FETCH_MODELS)}
          </p>
        )}
      </fieldset>
    </div>
  );
}
