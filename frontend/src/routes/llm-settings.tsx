import React from "react";
import { useTranslation } from "react-i18next";
import { AxiosError } from "axios";
import { useSearchParams } from "react-router";
import { ModelSelector } from "#/components/shared/modals/settings/model-selector";
import { organizeModelsAndProviders } from "#/utils/organize-models-and-providers";
import { useAIConfigOptions } from "#/hooks/query/use-ai-config-options";
import { useSettings } from "#/hooks/query/use-settings";
import { hasAdvancedSettingsSet } from "#/utils/has-advanced-settings-set";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { TooltipButton } from "#/components/shared/buttons/tooltip-button";
import QuestionCircleIcon from "#/icons/question-circle.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { HelpLink } from "#/ui/help-link";
import { BrandButton } from "#/components/features/settings/brand-button";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { useConfig } from "#/hooks/query/use-config";
import { isCustomModel } from "#/utils/is-custom-model";
import { LlmSettingsInputsSkeleton } from "#/components/features/settings/llm-settings/llm-settings-inputs-skeleton";
import { KeyStatusIcon } from "#/components/features/settings/key-status-icon";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { getProviderId } from "#/utils/map-provider";
import { DEFAULT_OPENHANDS_MODEL } from "#/utils/verified-models";

interface OpenHandsApiKeyHelpProps {
  testId: string;
}

function OpenHandsApiKeyHelp({ testId }: OpenHandsApiKeyHelpProps) {
  const { t } = useTranslation();

  return (
    <>
      <HelpLink
        testId={testId}
        text={t(I18nKey.SETTINGS$OPENHANDS_API_KEY_HELP_TEXT)}
        linkText={t(I18nKey.SETTINGS$NAV_API_KEYS)}
        href="https://app.all-hands.dev/settings/api-keys"
        suffix={` ${t(I18nKey.SETTINGS$OPENHANDS_API_KEY_HELP_SUFFIX)}`}
      />
      <p className="text-xs">
        {t(I18nKey.SETTINGS$LLM_BILLING_INFO)}{" "}
        <a
          href="https://docs.all-hands.dev/usage/llms/openhands-llms"
          rel="noreferrer noopener"
          target="_blank"
          className="underline underline-offset-2"
        >
          {t(I18nKey.SETTINGS$SEE_PRICING_DETAILS)}
        </a>
      </p>
    </>
  );
}

function LlmSettingsScreen() {
  const { t } = useTranslation();
  const [searchParams, setSearchParams] = useSearchParams();

  const { mutate: saveSettings, isPending } = useSaveSettings();

  const { data: resources } = useAIConfigOptions();
  const { data: settings, isLoading, isFetching } = useSettings();
  const { data: config } = useConfig();

  const [view, setView] = React.useState<"basic" | "advanced">("basic");

  const [dirtyInputs, setDirtyInputs] = React.useState({
    model: false,
    apiKey: false,
    searchApiKey: false,
    baseUrl: false,
    agent: false,
    confirmationMode: false,
    enableDefaultCondenser: false,
    securityAnalyzer: false,
    condenserMaxSize: false,
  });

  // Track the currently selected model to show help text
  const [currentSelectedModel, setCurrentSelectedModel] = React.useState<
    string | null
  >(null);

  // Track confirmation mode state to control security analyzer visibility
  const [confirmationModeEnabled, setConfirmationModeEnabled] = React.useState(
    settings?.CONFIRMATION_MODE ?? DEFAULT_SETTINGS.CONFIRMATION_MODE,
  );

  // Track selected security analyzer for form submission
  const [selectedSecurityAnalyzer, setSelectedSecurityAnalyzer] =
    React.useState(
      settings?.SECURITY_ANALYZER === null
        ? "none"
        : (settings?.SECURITY_ANALYZER ?? DEFAULT_SETTINGS.SECURITY_ANALYZER),
    );

  const [selectedProvider, setSelectedProvider] = React.useState<string | null>(
    null,
  );

  const modelsAndProviders = organizeModelsAndProviders(
    resources?.models || [],
  );

  // Determine if we should hide the API key input and use OpenHands-managed key (when using OpenHands provider in SaaS mode)
  const currentModel = currentSelectedModel || settings?.LLM_MODEL;

  const isSaasMode = config?.APP_MODE === "saas";

  const isOpenHandsProvider = () => {
    if (view === "basic") {
      return selectedProvider === "openhands";
    }

    if (view === "advanced") {
      if (dirtyInputs.model) {
        return currentModel?.startsWith("openhands/");
      }
      return settings?.LLM_MODEL?.startsWith("openhands/");
    }

    return false;
  };

  const shouldUseOpenHandsKey = isOpenHandsProvider() && isSaasMode;

  // Determine if we should hide the agent dropdown when V1 conversation API is enabled
  const isV1Enabled = settings?.V1_ENABLED;

  React.useEffect(() => {
    const determineWhetherToToggleAdvancedSettings = () => {
      if (resources && settings) {
        return (
          isCustomModel(resources.models, settings.LLM_MODEL) ||
          hasAdvancedSettingsSet({
            ...settings,
          })
        );
      }

      return false;
    };

    const userSettingsIsAdvanced = determineWhetherToToggleAdvancedSettings();

    if (userSettingsIsAdvanced) setView("advanced");
    else setView("basic");
  }, [settings, resources]);

  // Initialize currentSelectedModel with the current settings
  React.useEffect(() => {
    if (settings?.LLM_MODEL) {
      setCurrentSelectedModel(settings.LLM_MODEL);
    }
  }, [settings?.LLM_MODEL]);

  // Update confirmation mode state when settings change
  React.useEffect(() => {
    if (settings?.CONFIRMATION_MODE !== undefined) {
      setConfirmationModeEnabled(settings.CONFIRMATION_MODE);
    }
  }, [settings?.CONFIRMATION_MODE]);

  // Update selected security analyzer state when settings change
  React.useEffect(() => {
    if (settings?.SECURITY_ANALYZER !== undefined) {
      setSelectedSecurityAnalyzer(settings.SECURITY_ANALYZER || "none");
    }
  }, [settings?.SECURITY_ANALYZER]);

  // Handle URL parameters for SaaS subscription redirects
  React.useEffect(() => {
    const checkout = searchParams.get("checkout");

    if (checkout === "success") {
      displaySuccessToast(t(I18nKey.SUBSCRIPTION$SUCCESS));
      setSearchParams({});
    } else if (checkout === "cancel") {
      displayErrorToast(t(I18nKey.SUBSCRIPTION$FAILURE));
      setSearchParams({});
    }
  }, [searchParams, setSearchParams, t]);

  const handleSuccessfulMutation = () => {
    displaySuccessToast(t(I18nKey.SETTINGS$SAVED_WARNING));
    setDirtyInputs({
      model: false,
      apiKey: false,
      searchApiKey: false,
      baseUrl: false,
      agent: false,
      confirmationMode: false,
      enableDefaultCondenser: false,
      securityAnalyzer: false,
      condenserMaxSize: false,
    });
  };

  const handleErrorMutation = (error: AxiosError) => {
    const errorMessage = retrieveAxiosErrorMessage(error);
    displayErrorToast(errorMessage || t(I18nKey.ERROR$GENERIC));
  };

  const basicFormAction = (formData: FormData) => {
    const providerDisplay = formData.get("llm-provider-input")?.toString();
    const provider = providerDisplay
      ? getProviderId(providerDisplay)
      : undefined;
    const model = formData.get("llm-model-input")?.toString();
    const apiKey = formData.get("llm-api-key-input")?.toString();
    const searchApiKey = formData.get("search-api-key-input")?.toString();
    const confirmationMode =
      formData.get("enable-confirmation-mode-switch")?.toString() === "on";
    const securityAnalyzer = formData
      .get("security-analyzer-input")
      ?.toString();

    const fullLlmModel = provider && model && `${provider}/${model}`;

    // Use OpenHands-managed key for OpenHands provider in SaaS mode
    const finalApiKey = shouldUseOpenHandsKey ? null : apiKey;

    saveSettings(
      {
        LLM_MODEL: fullLlmModel,
        llm_api_key: finalApiKey || null,
        SEARCH_API_KEY: searchApiKey || "",
        CONFIRMATION_MODE: confirmationMode,
        SECURITY_ANALYZER:
          securityAnalyzer === "none"
            ? null
            : securityAnalyzer || DEFAULT_SETTINGS.SECURITY_ANALYZER,

        // reset advanced settings
        LLM_BASE_URL: DEFAULT_SETTINGS.LLM_BASE_URL,
        AGENT: DEFAULT_SETTINGS.AGENT,
        ENABLE_DEFAULT_CONDENSER: DEFAULT_SETTINGS.ENABLE_DEFAULT_CONDENSER,
      },
      {
        onSuccess: handleSuccessfulMutation,
        onError: handleErrorMutation,
      },
    );
  };

  const advancedFormAction = (formData: FormData) => {
    const model = formData.get("llm-custom-model-input")?.toString();
    const baseUrl = formData.get("base-url-input")?.toString();
    const apiKey = formData.get("llm-api-key-input")?.toString();
    const searchApiKey = formData.get("search-api-key-input")?.toString();
    const agent = formData.get("agent-input")?.toString();
    const confirmationMode =
      formData.get("enable-confirmation-mode-switch")?.toString() === "on";
    const enableDefaultCondenser =
      formData.get("enable-memory-condenser-switch")?.toString() === "on";
    const condenserMaxSizeStr = formData
      .get("condenser-max-size-input")
      ?.toString();
    const condenserMaxSizeRaw = condenserMaxSizeStr
      ? Number.parseInt(condenserMaxSizeStr, 10)
      : undefined;
    const condenserMaxSize =
      condenserMaxSizeRaw !== undefined
        ? Math.max(20, condenserMaxSizeRaw)
        : undefined;

    const securityAnalyzer = formData
      .get("security-analyzer-input")
      ?.toString();

    // Use OpenHands-managed key for OpenHands provider in SaaS mode
    const finalApiKey = shouldUseOpenHandsKey ? null : apiKey;

    saveSettings(
      {
        LLM_MODEL: model,
        LLM_BASE_URL: baseUrl,
        llm_api_key: finalApiKey || null,
        SEARCH_API_KEY: searchApiKey || "",
        AGENT: agent,
        CONFIRMATION_MODE: confirmationMode,
        ENABLE_DEFAULT_CONDENSER: enableDefaultCondenser,
        CONDENSER_MAX_SIZE:
          condenserMaxSize ?? DEFAULT_SETTINGS.CONDENSER_MAX_SIZE,
        SECURITY_ANALYZER:
          securityAnalyzer === "none"
            ? null
            : securityAnalyzer || DEFAULT_SETTINGS.SECURITY_ANALYZER,
      },
      {
        onSuccess: handleSuccessfulMutation,
        onError: handleErrorMutation,
      },
    );
  };

  const handleToggleAdvancedSettings = (isToggled: boolean) => {
    setView(isToggled ? "advanced" : "basic");
    setDirtyInputs({
      model: false,
      apiKey: false,
      searchApiKey: false,
      baseUrl: false,
      agent: false,
      confirmationMode: false,
      enableDefaultCondenser: false,
      securityAnalyzer: false,
      condenserMaxSize: false,
    });
  };

  const handleModelIsDirty = (
    provider: string | null,
    model: string | null,
  ) => {
    // openai providers are special case; see ModelSelector
    // component for details
    const modelIsDirty = model !== settings?.LLM_MODEL.replace("openai/", "");
    setDirtyInputs((prev) => ({
      ...prev,
      model: modelIsDirty,
    }));

    // Track the currently selected model for help text display
    setCurrentSelectedModel(model);
    setSelectedProvider(provider);
  };

  const onDefaultValuesChanged = (
    provider: string | null,
    model: string | null,
  ) => {
    setSelectedProvider(provider);
    setCurrentSelectedModel(model);
  };

  const handleApiKeyIsDirty = (apiKey: string) => {
    const apiKeyIsDirty = apiKey !== "";
    setDirtyInputs((prev) => ({
      ...prev,
      apiKey: apiKeyIsDirty,
    }));
  };

  const handleSearchApiKeyIsDirty = (searchApiKey: string) => {
    const searchApiKeyIsDirty = searchApiKey !== settings?.SEARCH_API_KEY;
    setDirtyInputs((prev) => ({
      ...prev,
      searchApiKey: searchApiKeyIsDirty,
    }));
  };

  const handleCustomModelIsDirty = (model: string) => {
    const modelIsDirty = model !== settings?.LLM_MODEL && model !== "";
    setDirtyInputs((prev) => ({
      ...prev,
      model: modelIsDirty,
    }));

    // Track the currently selected model for help text display
    setCurrentSelectedModel(model);
  };

  const handleBaseUrlIsDirty = (baseUrl: string) => {
    const baseUrlIsDirty = baseUrl !== settings?.LLM_BASE_URL;
    setDirtyInputs((prev) => ({
      ...prev,
      baseUrl: baseUrlIsDirty,
    }));
  };

  const handleAgentIsDirty = (agent: string) => {
    const agentIsDirty = agent !== settings?.AGENT && agent !== "";
    setDirtyInputs((prev) => ({
      ...prev,
      agent: agentIsDirty,
    }));
  };

  const handleConfirmationModeIsDirty = (isToggled: boolean) => {
    const confirmationModeIsDirty = isToggled !== settings?.CONFIRMATION_MODE;
    setDirtyInputs((prev) => ({
      ...prev,
      confirmationMode: confirmationModeIsDirty,
    }));
    setConfirmationModeEnabled(isToggled);

    // When confirmation mode is enabled, set default security analyzer to "llm" if not already set
    if (isToggled && !selectedSecurityAnalyzer) {
      setSelectedSecurityAnalyzer(DEFAULT_SETTINGS.SECURITY_ANALYZER);
      setDirtyInputs((prev) => ({
        ...prev,
        securityAnalyzer: true,
      }));
    }
  };

  const handleEnableDefaultCondenserIsDirty = (isToggled: boolean) => {
    const enableDefaultCondenserIsDirty =
      isToggled !== settings?.ENABLE_DEFAULT_CONDENSER;
    setDirtyInputs((prev) => ({
      ...prev,
      enableDefaultCondenser: enableDefaultCondenserIsDirty,
    }));
  };

  const handleCondenserMaxSizeIsDirty = (value: string) => {
    const parsed = value ? Number.parseInt(value, 10) : undefined;
    const bounded = parsed !== undefined ? Math.max(20, parsed) : undefined;
    const condenserMaxSizeIsDirty =
      (bounded ?? DEFAULT_SETTINGS.CONDENSER_MAX_SIZE) !==
      (settings?.CONDENSER_MAX_SIZE ?? DEFAULT_SETTINGS.CONDENSER_MAX_SIZE);
    setDirtyInputs((prev) => ({
      ...prev,
      condenserMaxSize: condenserMaxSizeIsDirty,
    }));
  };

  const handleSecurityAnalyzerIsDirty = (securityAnalyzer: string) => {
    const securityAnalyzerIsDirty =
      securityAnalyzer !== settings?.SECURITY_ANALYZER;
    setDirtyInputs((prev) => ({
      ...prev,
      securityAnalyzer: securityAnalyzerIsDirty,
    }));
  };

  const formIsDirty = Object.values(dirtyInputs).some((isDirty) => isDirty);

  const getSecurityAnalyzerOptions = () => {
    const analyzers = resources?.securityAnalyzers || [];
    const orderedItems = [];

    // Add LLM analyzer first
    if (analyzers.includes("llm")) {
      orderedItems.push({
        key: "llm",
        label: t(I18nKey.SETTINGS$SECURITY_ANALYZER_LLM_DEFAULT),
      });
    }

    // Add None option second
    orderedItems.push({
      key: "none",
      label: t(I18nKey.SETTINGS$SECURITY_ANALYZER_NONE),
    });

    if (isV1Enabled) {
      return orderedItems;
    }

    // Add Invariant analyzer third
    if (analyzers.includes("invariant")) {
      orderedItems.push({
        key: "invariant",
        label: t(I18nKey.SETTINGS$SECURITY_ANALYZER_INVARIANT),
      });
    }

    // Add any other analyzers that might exist
    analyzers.forEach((analyzer) => {
      if (!["llm", "invariant", "none"].includes(analyzer)) {
        // For unknown analyzers, use the analyzer name as fallback
        // In the future, add specific i18n keys for new analyzers
        orderedItems.push({
          key: analyzer,
          label: analyzer, // TODO: Add i18n support for new analyzers
        });
      }
    });

    return orderedItems;
  };

  if (!settings || isFetching) return <LlmSettingsInputsSkeleton />;

  const formAction = (formData: FormData) => {
    if (view === "basic") basicFormAction(formData);
    else advancedFormAction(formData);
  };

  return (
    <div data-testid="llm-settings-screen" className="h-full relative">
      <form
        action={formAction}
        className="flex flex-col h-full justify-between"
      >
        <div className="flex flex-col gap-6">
          <SettingsSwitch
            testId="advanced-settings-switch"
            defaultIsToggled={view === "advanced"}
            onToggle={handleToggleAdvancedSettings}
            isToggled={view === "advanced"}
          >
            {t(I18nKey.SETTINGS$ADVANCED)}
          </SettingsSwitch>

          {view === "basic" && (
            <div
              data-testid="llm-settings-form-basic"
              className="flex flex-col gap-6"
            >
              {!isLoading && !isFetching && (
                <>
                  <ModelSelector
                    models={modelsAndProviders}
                    currentModel={settings.LLM_MODEL || DEFAULT_OPENHANDS_MODEL}
                    onChange={handleModelIsDirty}
                    onDefaultValuesChanged={onDefaultValuesChanged}
                    wrapperClassName="!flex-col !gap-6"
                  />
                  {(settings.LLM_MODEL?.startsWith("openhands/") ||
                    currentSelectedModel?.startsWith("openhands/")) && (
                    <OpenHandsApiKeyHelp testId="openhands-api-key-help" />
                  )}
                </>
              )}

              {!shouldUseOpenHandsKey && (
                <>
                  <SettingsInput
                    testId="llm-api-key-input"
                    name="llm-api-key-input"
                    label={t(I18nKey.SETTINGS_FORM$API_KEY)}
                    type="password"
                    className="w-full max-w-[680px]"
                    placeholder={settings.LLM_API_KEY_SET ? "<hidden>" : ""}
                    onChange={handleApiKeyIsDirty}
                    startContent={
                      settings.LLM_API_KEY_SET && (
                        <KeyStatusIcon isSet={settings.LLM_API_KEY_SET} />
                      )
                    }
                  />

                  <HelpLink
                    testId="llm-api-key-help-anchor"
                    text={t(I18nKey.SETTINGS$DONT_KNOW_API_KEY)}
                    linkText={t(I18nKey.SETTINGS$CLICK_FOR_INSTRUCTIONS)}
                    href="https://docs.all-hands.dev/usage/local-setup#getting-an-api-key"
                  />
                </>
              )}
            </div>
          )}

          {view === "advanced" && (
            <div
              data-testid="llm-settings-form-advanced"
              className="flex flex-col gap-6"
            >
              <SettingsInput
                testId="llm-custom-model-input"
                name="llm-custom-model-input"
                label={t(I18nKey.SETTINGS$CUSTOM_MODEL)}
                defaultValue={settings.LLM_MODEL || DEFAULT_OPENHANDS_MODEL}
                placeholder={DEFAULT_OPENHANDS_MODEL}
                type="text"
                className="w-full max-w-[680px]"
                onChange={handleCustomModelIsDirty}
              />
              {(settings.LLM_MODEL?.startsWith("openhands/") ||
                currentSelectedModel?.startsWith("openhands/")) && (
                <OpenHandsApiKeyHelp testId="openhands-api-key-help-2" />
              )}

              <SettingsInput
                testId="base-url-input"
                name="base-url-input"
                label={t(I18nKey.SETTINGS$BASE_URL)}
                defaultValue={settings.LLM_BASE_URL}
                placeholder="https://api.openai.com"
                type="text"
                className="w-full max-w-[680px]"
                onChange={handleBaseUrlIsDirty}
              />

              {!shouldUseOpenHandsKey && (
                <>
                  <SettingsInput
                    testId="llm-api-key-input"
                    name="llm-api-key-input"
                    label={t(I18nKey.SETTINGS_FORM$API_KEY)}
                    type="password"
                    className="w-full max-w-[680px]"
                    placeholder={settings.LLM_API_KEY_SET ? "<hidden>" : ""}
                    onChange={handleApiKeyIsDirty}
                    startContent={
                      settings.LLM_API_KEY_SET && (
                        <KeyStatusIcon isSet={settings.LLM_API_KEY_SET} />
                      )
                    }
                  />
                  <HelpLink
                    testId="llm-api-key-help-anchor-advanced"
                    text={t(I18nKey.SETTINGS$DONT_KNOW_API_KEY)}
                    linkText={t(I18nKey.SETTINGS$CLICK_FOR_INSTRUCTIONS)}
                    href="https://docs.all-hands.dev/usage/local-setup#getting-an-api-key"
                  />
                </>
              )}

              {config?.APP_MODE !== "saas" && (
                <>
                  <SettingsInput
                    testId="search-api-key-input"
                    name="search-api-key-input"
                    label={t(I18nKey.SETTINGS$SEARCH_API_KEY)}
                    type="password"
                    className="w-full max-w-[680px]"
                    defaultValue={settings.SEARCH_API_KEY || ""}
                    onChange={handleSearchApiKeyIsDirty}
                    placeholder={t(I18nKey.API$TVLY_KEY_EXAMPLE)}
                    startContent={
                      settings.SEARCH_API_KEY_SET && (
                        <KeyStatusIcon isSet={settings.SEARCH_API_KEY_SET} />
                      )
                    }
                  />

                  <HelpLink
                    testId="search-api-key-help-anchor"
                    text={t(I18nKey.SETTINGS$SEARCH_API_KEY_OPTIONAL)}
                    linkText={t(I18nKey.SETTINGS$SEARCH_API_KEY_INSTRUCTIONS)}
                    href="https://tavily.com/"
                  />

                  {!isV1Enabled && (
                    <SettingsDropdownInput
                      testId="agent-input"
                      name="agent-input"
                      label={t(I18nKey.SETTINGS$AGENT)}
                      items={
                        resources?.agents.map((agent) => ({
                          key: agent,
                          label: agent, // TODO: Add i18n support for agent names
                        })) || []
                      }
                      defaultSelectedKey={settings.AGENT}
                      isClearable={false}
                      onInputChange={handleAgentIsDirty}
                      wrapperClassName="w-full max-w-[680px]"
                    />
                  )}
                </>
              )}

              <div className="w-full max-w-[680px]">
                <SettingsInput
                  testId="condenser-max-size-input"
                  name="condenser-max-size-input"
                  type="number"
                  min={20}
                  step={1}
                  label={t(I18nKey.SETTINGS$CONDENSER_MAX_SIZE)}
                  defaultValue={(
                    settings.CONDENSER_MAX_SIZE ??
                    DEFAULT_SETTINGS.CONDENSER_MAX_SIZE
                  )?.toString()}
                  onChange={(value) => handleCondenserMaxSizeIsDirty(value)}
                  isDisabled={!settings.ENABLE_DEFAULT_CONDENSER}
                />
                <p className="text-xs text-tertiary-alt mt-1">
                  {t(I18nKey.SETTINGS$CONDENSER_MAX_SIZE_TOOLTIP)}
                </p>
              </div>

              <SettingsSwitch
                testId="enable-memory-condenser-switch"
                name="enable-memory-condenser-switch"
                defaultIsToggled={settings.ENABLE_DEFAULT_CONDENSER}
                onToggle={handleEnableDefaultCondenserIsDirty}
              >
                {t(I18nKey.SETTINGS$ENABLE_MEMORY_CONDENSATION)}
              </SettingsSwitch>

              {/* Confirmation mode and security analyzer */}
              <div className="flex items-center gap-2">
                <SettingsSwitch
                  testId="enable-confirmation-mode-switch"
                  name="enable-confirmation-mode-switch"
                  onToggle={handleConfirmationModeIsDirty}
                  defaultIsToggled={settings.CONFIRMATION_MODE}
                  isBeta
                >
                  {t(I18nKey.SETTINGS$CONFIRMATION_MODE)}
                </SettingsSwitch>
                <TooltipButton
                  tooltip={t(I18nKey.SETTINGS$CONFIRMATION_MODE_TOOLTIP)}
                  ariaLabel={t(I18nKey.SETTINGS$CONFIRMATION_MODE)}
                  className="text-[#9099AC] hover:text-white cursor-help"
                >
                  <QuestionCircleIcon width={16} height={16} />
                </TooltipButton>
              </div>

              {confirmationModeEnabled && (
                <>
                  <div className="w-full max-w-[680px]">
                    <SettingsDropdownInput
                      testId="security-analyzer-input"
                      name="security-analyzer-display"
                      label={t(I18nKey.SETTINGS$SECURITY_ANALYZER)}
                      items={getSecurityAnalyzerOptions()}
                      placeholder={t(
                        I18nKey.SETTINGS$SECURITY_ANALYZER_PLACEHOLDER,
                      )}
                      selectedKey={selectedSecurityAnalyzer || "none"}
                      isClearable={false}
                      onSelectionChange={(key) => {
                        const newValue = key?.toString() || "";
                        setSelectedSecurityAnalyzer(newValue);
                        handleSecurityAnalyzerIsDirty(newValue);
                      }}
                      onInputChange={(value) => {
                        // Handle when input is cleared
                        if (!value) {
                          setSelectedSecurityAnalyzer("");
                          handleSecurityAnalyzerIsDirty("");
                        }
                      }}
                      wrapperClassName="w-full"
                    />
                    {/* Hidden input to store the actual key value for form submission */}
                    <input
                      type="hidden"
                      name="security-analyzer-input"
                      value={selectedSecurityAnalyzer || ""}
                    />
                  </div>
                  <p className="text-xs text-tertiary-alt max-w-[680px]">
                    {t(I18nKey.SETTINGS$SECURITY_ANALYZER_DESCRIPTION)}
                  </p>
                </>
              )}
            </div>
          )}
        </div>

        <div className="flex gap-6 p-6 justify-end">
          <BrandButton
            testId="submit-button"
            type="submit"
            variant="primary"
            isDisabled={!formIsDirty || isPending}
          >
            {!isPending && t("SETTINGS$SAVE_CHANGES")}
            {isPending && t("SETTINGS$SAVING")}
          </BrandButton>
        </div>
      </form>
    </div>
  );
}

export default LlmSettingsScreen;
