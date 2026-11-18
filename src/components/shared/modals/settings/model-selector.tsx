import { ComboBox, Header, Input, ListBox, Spinner } from "@heroui/react";
import React, { useContext } from "react";
import { ComboBoxStateContext } from "react-aria-components";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { mapProvider } from "#/utils/map-provider";
import { extractModelAndProvider } from "#/utils/extract-model-and-provider";
import { cn } from "#/utils/utils";
import { statusToneBadgeClassName } from "#/utils/status-tone-classes";
import { formControlSettingsFieldClassName } from "#/utils/form-control-classes";
import { heroUiAutocompleteSelectorButtonClassName } from "#/ui/combobox-caret";
import { HelpLink } from "#/ui/help-link";
import { PRODUCT_URL } from "#/utils/constants";
import { useSearchProviders } from "#/hooks/query/use-search-providers";
import { useProviderModels } from "#/hooks/query/use-provider-models";
import { FREE_MODEL_BADGE_LABEL } from "#/utils/format-model-name";
import { FreeOpenHandsModelsNote } from "#/components/shared/free-models-note";
import type { LLMProvider } from "#/api/config-service/config-service.types";

// Sorts by the label the picker shows, keeping the managed OpenHands provider
// first as the backend does.
const compareProviderLabels = (a: LLMProvider, b: LLMProvider) =>
  Number(b.name === "openhands") - Number(a.name === "openhands") ||
  mapProvider(a.name).localeCompare(mapProvider(b.name), undefined, {
    sensitivity: "base",
  });

const freeModelBadgeClassName = cn(
  "shrink-0 rounded-full border border-warning/40 px-1.5 py-0.5 text-3xs leading-none dark:bg-warning/10 dark:text-warning",
  statusToneBadgeClassName.warning,
);

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

function OpenOnClickInput({ testId, name }: { testId: string; name: string }) {
  const state = useContext(ComboBoxStateContext);
  return (
    <Input
      data-testid={testId}
      name={name}
      className="border-0 bg-transparent shadow-none outline-none focus:ring-0 focus-visible:ring-0 data-[focused=true]:ring-0"
      onClick={() => {
        if (state && !state.isOpen) state.open(null, "manual");
      }}
    />
  );
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
    () => providers.filter((p) => p.verified).sort(compareProviderLabels),
    [providers],
  );
  const unverifiedProviders = React.useMemo(
    () => providers.filter((p) => !p.verified).sort(compareProviderLabels),
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
    setSelectedModel(null);
    setLitellmId(null);
  };

  const isSelectedModelFree = Boolean(
    selectedModel && freeModelNameSet.has(selectedModel),
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
        "flex flex-col md:flex-row w-full min-w-0 justify-between gap-4 md:gap-11.5",
        wrapperClassName,
      )}
    >
      <fieldset className="flex flex-col gap-2.5 w-full">
        <label className={cn("text-sm", labelClassName)}>
          {t(I18nKey.LLM$PROVIDER)}
        </label>
        <ComboBox
          menuTrigger="focus"
          allowsEmptyCollection
          isRequired
          isDisabled={isDisabled}
          aria-label={t(I18nKey.LLM$PROVIDER)}
          onSelectionChange={(key) => {
            if (key?.toString()) handleChangeProvider(key.toString());
          }}
          onInputChange={(value) => !value && clear()}
          // Select the current provider's label so typing replaces it and
          // filters the list instead of appending to it.
          onFocus={(event) => {
            if (event.target instanceof HTMLInputElement) {
              event.target.select();
            }
          }}
          selectedKey={selectedProvider}
        >
          <ComboBox.InputGroup
            className={cn(
              formControlSettingsFieldClassName,
              "focus-within:border-contrast/40 dark:focus-within:border-contrast/40 focus-within:ring-1 focus-within:ring-contrast/20",
            )}
          >
            <OpenOnClickInput
              testId="llm-provider-input"
              name="llm-provider-input"
            />
            <ComboBox.Trigger
              className={heroUiAutocompleteSelectorButtonClassName}
            />
          </ComboBox.InputGroup>
          <ComboBox.Popover className="rounded-xl border border-border">
            <ListBox>
              <ListBox.Section>
                <Header className="text-muted">
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
                  <Header className="text-muted">
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
            menuTrigger="focus"
            allowsEmptyCollection
            isRequired
            aria-label={t(I18nKey.LLM$MODEL)}
            onSelectionChange={(key) => {
              if (key?.toString()) handleChangeModel(key.toString());
            }}
            isDisabled={isDisabled || !selectedProvider}
            selectedKey={selectedModel}
          >
            <ComboBox.InputGroup
              className={cn(
                formControlSettingsFieldClassName,
                "focus-within:border-contrast/40 dark:focus-within:border-contrast/40 focus-within:ring-1 focus-within:ring-contrast/20",
              )}
            >
              <OpenOnClickInput
                testId="llm-model-input"
                name="llm-model-input"
              />
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
            <ComboBox.Popover className="rounded-xl border border-border">
              <ListBox>
                <ListBox.Section>
                  <Header className="text-muted">
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
                        {model.free ? (
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
                    <Header className="text-muted">
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
