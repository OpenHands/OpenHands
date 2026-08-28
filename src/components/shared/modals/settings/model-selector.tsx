import { ComboBox, Header, Input, ListBox, Spinner } from "@heroui/react";
import { Info } from "lucide-react";
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
import {
  FREE_MODEL_BADGE_LABEL,
  FREE_OPENHANDS_MODEL_NOTE,
  isFreeOpenHandsModel,
} from "#/utils/format-model-name";

const freeModelBadgeClassName =
  "shrink-0 rounded-full border border-warning/40 bg-warning/10 px-1.5 py-0.5 text-[10px] leading-none text-warning";

function FreeOpenHandsModelsNote() {
  return (
    <p
      data-testid="openhands-free-models-note"
      className="flex items-start gap-2 text-xs text-warning"
    >
      <Info className="mt-0.5 size-4 shrink-0 text-warning" aria-hidden />
      <span>{FREE_OPENHANDS_MODEL_NOTE}</span>
    </p>
  );
}

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

  const selectedFullModel =
    selectedProvider && selectedModel
      ? `${selectedProvider}/${selectedModel}`
      : null;
  const isSelectedModelFree = isFreeOpenHandsModel(selectedFullModel);
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
        <ComboBox
          isRequired
          isDisabled={isDisabled}
          aria-label={t(I18nKey.LLM$PROVIDER)}
          onSelectionChange={(e) => {
            if (e?.toString()) handleChangeProvider(e.toString());
          }}
          onInputChange={(value) => !value && clear()}
          defaultSelectedKey={selectedProvider ?? undefined}
          selectedKey={selectedProvider}
        >
          <ComboBox.InputGroup className={formControlSettingsFieldClassName}>
            <Input data-testid="llm-provider-input" name="llm-provider-input" />
            <ComboBox.Trigger
              className={heroUiAutocompleteSelectorButtonClassName}
            />
          </ComboBox.InputGroup>
          <ComboBox.Popover className="rounded-xl border border-[var(--oh-border)]">
            <ListBox>
              <ListBox.Section>
                <Header className="text-[var(--oh-muted)]">
                  {t(I18nKey.MODEL_SELECTOR$VERIFIED)}
                </Header>
                {verifiedProviders.map((provider) => (
                  <ListBox.Item
                    data-testid={`provider-item-${provider.name}`}
                    id={provider.name}
                    key={provider.name}
                    textValue={mapProvider(provider.name)}
                  >
                    {mapProvider(provider.name)}
                  </ListBox.Item>
                ))}
              </ListBox.Section>
              {unverifiedProviders.length > 0 ? (
                <ListBox.Section>
                  <Header className="text-[var(--oh-muted)]">
                    {t(I18nKey.MODEL_SELECTOR$OTHERS)}
                  </Header>
                  {unverifiedProviders.map((provider) => (
                    <ListBox.Item
                      id={provider.name}
                      key={provider.name}
                      textValue={mapProvider(provider.name)}
                    >
                      {mapProvider(provider.name)}
                    </ListBox.Item>
                  ))}
                </ListBox.Section>
              ) : null}
            </ListBox>
          </ComboBox.Popover>
        </ComboBox>
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
          <ComboBox
            isRequired
            aria-label={t(I18nKey.LLM$MODEL)}
            onSelectionChange={(e) => {
              if (e?.toString()) handleChangeModel(e.toString());
            }}
            isDisabled={isDisabled || !selectedProvider}
            selectedKey={selectedModel}
            defaultSelectedKey={selectedModel ?? undefined}
          >
            <ComboBox.InputGroup className={formControlSettingsFieldClassName}>
              <Input data-testid="llm-model-input" name="llm-model-input" />
              {isLoadingModels ? (
                <Spinner
                  size="sm"
                  className="mr-2"
                  aria-label={t(I18nKey.HOME$LOADING)}
                />
              ) : null}
              <ComboBox.Trigger
                className={heroUiAutocompleteSelectorButtonClassName}
              />
            </ComboBox.InputGroup>
            <ComboBox.Popover className="rounded-xl border border-[var(--oh-border)]">
              <ListBox>
                <ListBox.Section>
                  <Header className="text-[var(--oh-muted)]">
                    {t(I18nKey.MODEL_SELECTOR$VERIFIED)}
                  </Header>
                  {verifiedModels.map((model) => (
                    <ListBox.Item
                      key={model.name}
                      id={model.name}
                      textValue={model.name}
                    >
                      <span className="flex min-w-0 items-center gap-2">
                        <span className="truncate">{model.name}</span>
                        {isFreeOpenHandsModel(
                          `${selectedProvider}/${model.name}`,
                        ) ? (
                          <span className={freeModelBadgeClassName}>
                            {FREE_MODEL_BADGE_LABEL}
                          </span>
                        ) : null}
                      </span>
                    </ListBox.Item>
                  ))}
                </ListBox.Section>
                {unverifiedModels.length > 0 ? (
                  <ListBox.Section>
                    <Header className="text-[var(--oh-muted)]">
                      {t(I18nKey.MODEL_SELECTOR$OTHERS)}
                    </Header>
                    {unverifiedModels.map((model) => (
                      <ListBox.Item
                        data-testid={`model-item-${model.name}`}
                        key={model.name}
                        id={model.name}
                        textValue={model.name}
                      >
                        {model.name}
                      </ListBox.Item>
                    ))}
                  </ListBox.Section>
                ) : null}
              </ListBox>
            </ComboBox.Popover>
          </ComboBox>
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
        {modelsError && (
          <p data-testid="models-error" className="text-danger text-xs">
            {t(I18nKey.CONFIGURATION$ERROR_FETCH_MODELS)}
          </p>
        )}
        {selectedProvider === "openhands" ? <FreeOpenHandsModelsNote /> : null}
      </fieldset>
    </div>
  );
}
