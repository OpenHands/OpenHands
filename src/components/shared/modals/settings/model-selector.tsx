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
import { OPENROUTER_PROVIDER } from "#/api/openrouter-models-service";
import type { LLMModel } from "#/api/config-service/config-service.types";
import { formatCompactTokenCount } from "#/utils/format-token-count";

const freeModelBadgeClassName =
  "shrink-0 rounded-full border border-warning/40 bg-warning/10 px-1.5 py-0.5 text-[10px] leading-none text-warning";

// "tool_choice" can accompany "tools" but doesn't imply it on its own — only
// "tools" itself means the model actually accepts tool/function definitions.
const TOOL_CAPABLE_PARAMETER_NAMES = ["tools"];
const REASONING_CAPABLE_PARAMETER_NAMES = [
  "reasoning",
  "include_reasoning",
  "reasoning_effort",
];

/**
 * Known-unsupported (not merely unconfirmed) tool calling. `supportedParameters`
 * absent means OpenRouter didn't publish the list at all — treated as unknown,
 * never as unsupported, so it stays visible in the default filtered view. An
 * explicit empty list (`[]`) is a known value and correctly falls through to
 * "incapable" below, since it contains none of the capable names.
 */
function isKnownToolIncapable(model: LLMModel): boolean {
  const params = model.capabilities?.supportedParameters;
  if (!params) return false;
  return !TOOL_CAPABLE_PARAMETER_NAMES.some((name) => params.includes(name));
}

function hasCapability(model: LLMModel, names: string[]): boolean {
  const params = model.capabilities?.supportedParameters;
  return Boolean(params && names.some((name) => params.includes(name)));
}

/**
 * Per-million-token price, or `null` when the raw string isn't a genuine
 * price. OpenRouter uses a `-1` sentinel on variable/"auto"-routed models
 * (e.g. `openrouter/auto`) to mean "varies, not a fixed price" — never render
 * that as a literal negative dollar amount. Zero is a valid (free) price and
 * is not rejected here; whether a model is "free" for OpenHands purposes is a
 * separate, OpenHands-specific flag, never derived from this display value.
 */
function formatPerMillionPrice(raw: string | undefined): string | null {
  const trimmed = raw?.trim();
  // `Number("   ")` coerces to `0`, not `NaN` — trim first so a whitespace-only
  // (or empty) string is treated as missing, never as a genuine free price.
  if (!trimmed) return null;
  const perToken = Number(trimmed);
  if (!Number.isFinite(perToken) || perToken < 0) return null;
  return `$${(perToken * 1_000_000).toFixed(2)}`;
}

interface ModelSelectorProps {
  isDisabled?: boolean;
  currentModel?: string;
  /**
   * When set, the provider is fixed (e.g. bound to a shared provider
   * connection): the provider combobox is disabled and provider selection /
   * model fetching use this value even before `currentModel` has a model.
   * Changing to a different fixed provider while `currentModel` is empty
   * clears any stale selected model; re-rendering with the *same*
   * `fixedProvider` never resets the model.
   */
  fixedProvider?: string;
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
  fixedProvider,
  onChange,
  onDefaultValuesChanged,
  wrapperClassName,
  labelClassName,
}: ModelSelectorProps) {
  const [, setLitellmId] = React.useState<string | null>(null);
  const [selectedProvider, setSelectedProvider] = React.useState<string | null>(
    fixedProvider ?? null,
  );
  const [selectedModel, setSelectedModel] = React.useState<string | null>(null);
  const [showAllOpenRouterModels, setShowAllOpenRouterModels] =
    React.useState(false);

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
  const unverifiedProviders = React.useMemo(() => {
    const unverified = providers.filter((p) => !p.verified);
    if (
      selectedProvider &&
      !providers.some((p) => p.name === selectedProvider)
    ) {
      return [...unverified, { name: selectedProvider, verified: false }];
    }
    return unverified;
  }, [providers, selectedProvider]);

  const isOpenRouterSelected = selectedProvider === OPENROUTER_PROVIDER;

  // Default OpenRouter to tool-capable models: a coding agent needs function
  // calling, and OpenRouter's catalog otherwise includes many models that
  // can't do it. A model with no published `supported_parameters` at all is
  // unknown, not known-unsupported, so it stays visible by default. The
  // already-selected model is never hidden by this filter — the picker never
  // enforces the catalog against a value the caller (or a prior save)
  // already chose. `showAllOpenRouterModels` is the explicit escape hatch.
  const toolFilteredModels = React.useMemo(() => {
    if (!isOpenRouterSelected || showAllOpenRouterModels) return providerModels;
    return providerModels.filter(
      (m) => m.name === selectedModel || !isKnownToolIncapable(m),
    );
  }, [
    providerModels,
    isOpenRouterSelected,
    showAllOpenRouterModels,
    selectedModel,
  ]);

  const verifiedModels = React.useMemo(
    () => toolFilteredModels.filter((m) => m.verified),
    [toolFilteredModels],
  );
  const unverifiedModels = React.useMemo(
    () => toolFilteredModels.filter((m) => !m.verified),
    [toolFilteredModels],
  );

  // DB-driven set of free model names for the selected provider. Mirrors the
  // `verified` flag: the frontend no longer hardcodes which models are free.
  const freeModelNames = React.useMemo(
    () => toolFilteredModels.filter((m) => m.free).map((m) => m.name),
    [toolFilteredModels],
  );
  const freeModelNameSet = React.useMemo(
    () => new Set(freeModelNames),
    [freeModelNames],
  );

  const selectedModelData = React.useMemo(
    () => providerModels.find((m) => m.name === selectedModel) ?? null,
    [providerModels, selectedModel],
  );
  const selectedModelCapabilities = selectedModelData?.capabilities;
  const selectedModelSupportsTools = Boolean(
    selectedModelData &&
    hasCapability(selectedModelData, TOOL_CAPABLE_PARAMETER_NAMES),
  );
  const selectedModelSupportsReasoning = Boolean(
    selectedModelData &&
    hasCapability(selectedModelData, REASONING_CAPABLE_PARAMETER_NAMES),
  );
  const selectedModelPromptPrice = formatPerMillionPrice(
    selectedModelCapabilities?.pricing?.prompt,
  );
  const selectedModelCompletionPrice = formatPerMillionPrice(
    selectedModelCapabilities?.pricing?.completion,
  );

  React.useEffect(() => {
    if (currentModel) {
      const { provider, model } = extractModelAndProvider(currentModel);
      const effectiveProvider = fixedProvider ?? provider ?? null;

      setLitellmId(currentModel);
      setSelectedProvider(effectiveProvider);
      setSelectedModel(model);
      onDefaultValuesChanged?.(provider || null, model);
    } else if (fixedProvider) {
      // The caller cleared the model (e.g. switching to a connection whose
      // fixed provider makes the previous model incompatible) — keep the
      // fixed provider but drop the now-stale model rather than keep
      // showing it. Re-renders with the *same* fixedProvider and no
      // currentModel change don't reach here (deps unchanged), so a
      // same-provider connection swap never resets a model the caller
      // intentionally kept.
      setSelectedProvider(fixedProvider);
      setSelectedModel(null);
      setLitellmId(null);
    }
  }, [currentModel, fixedProvider]);

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
          isDisabled={isDisabled || Boolean(fixedProvider)}
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
            name="llm-model-input"
            aria-label={t(I18nKey.LLM$MODEL)}
            isClearable={false}
            onSelectionChange={(e) => {
              if (e?.toString()) handleChangeModel(e.toString());
            }}
            isDisabled={isDisabled || !selectedProvider}
            selectedKey={selectedModel}
            defaultSelectedKey={selectedModel ?? undefined}
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
        {modelsError && (
          <p data-testid="models-error" className="text-danger text-xs">
            {t(I18nKey.CONFIGURATION$ERROR_FETCH_MODELS)}
          </p>
        )}
        {isOpenRouterSelected ? (
          <label
            data-testid="openrouter-show-all-models-toggle"
            className="flex items-center gap-2 text-xs text-[var(--oh-muted)]"
          >
            <input
              type="checkbox"
              checked={showAllOpenRouterModels}
              onChange={(e) => setShowAllOpenRouterModels(e.target.checked)}
            />
            {t(I18nKey.MODEL_SELECTOR$SHOW_ALL_OPENROUTER_MODELS)}
          </label>
        ) : null}
        {selectedModelCapabilities ? (
          <div
            data-testid="model-capabilities"
            className="flex flex-wrap gap-3 text-xs text-[var(--oh-muted)]"
          >
            {selectedModelCapabilities.contextLength ? (
              <span data-testid="model-context-length">
                {t(I18nKey.MODEL_SELECTOR$CONTEXT_LENGTH, {
                  value: formatCompactTokenCount(
                    selectedModelCapabilities.contextLength,
                  ),
                })}
              </span>
            ) : null}
            {selectedModelCapabilities.maxOutputTokens ? (
              <span data-testid="model-max-output">
                {t(I18nKey.MODEL_SELECTOR$MAX_OUTPUT, {
                  value: formatCompactTokenCount(
                    selectedModelCapabilities.maxOutputTokens,
                  ),
                })}
              </span>
            ) : null}
            {selectedModelSupportsTools ? (
              <span data-testid="model-supports-tools">
                {t(I18nKey.MODEL_SELECTOR$SUPPORTS_TOOLS)}
              </span>
            ) : null}
            {selectedModelSupportsReasoning ? (
              <span data-testid="model-supports-reasoning">
                {t(I18nKey.MODEL_SELECTOR$SUPPORTS_REASONING)}
              </span>
            ) : null}
            {selectedModelPromptPrice && selectedModelCompletionPrice ? (
              <span data-testid="model-pricing">
                {t(I18nKey.MODEL_SELECTOR$PRICING, {
                  prompt: selectedModelPromptPrice,
                  completion: selectedModelCompletionPrice,
                })}
              </span>
            ) : null}
          </div>
        ) : null}
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
