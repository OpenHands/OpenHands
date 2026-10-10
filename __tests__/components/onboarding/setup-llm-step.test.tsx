import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  ONBOARDING_DEFAULT_LLM_MODEL,
  SetupLlmStep,
} from "#/components/features/onboarding/steps/setup-llm-step";
import type { SdkSectionSaveControl } from "#/components/features/settings/sdk-settings/sdk-section-page";
import type { SettingsFormValues } from "#/utils/sdk-settings-schema";
import { renderWithProviders } from "../../../test-utils";

const saveProfile = vi.hoisted(() => vi.fn());
const activateProfile = vi.hoisted(() => vi.fn());
const listProfiles = vi.hoisted(() => vi.fn());
const getProfile = vi.hoisted(() => vi.fn());
const applyAgentProfile = vi.hoisted(() => vi.fn());
const listAgentProfiles = vi.hoisted(() => vi.fn());
const fetchSettings = vi.hoisted(() => vi.fn());
const settingsSave = vi.hoisted(() => vi.fn());
const displayErrorToast = vi.hoisted(() => vi.fn());

interface ScreenProps {
  initialValueOverrides?: SettingsFormValues;
  markInitialOverridesDirty?: boolean;
  onSaveSuccess?: () => void;
  onSaveControlChange: (control: SdkSectionSaveControl) => void;
}

const formState = vi.hoisted(() => ({
  backendKind: "local",
  backendId: "local-backend",
  /** What the user changed in the form, as the coerced `llm` payload. */
  dirtyLlm: {} as Record<string, unknown>,
  view: "all" as "basic" | "advanced" | "all",
  screenProps: undefined as ScreenProps | undefined,
  subscriptionModels: undefined as string[] | undefined,
}));

const llmField = (
  key: string,
  valueType: "string" | "number" = "string",
  extra: Record<string, unknown> = {},
) => ({
  key,
  label: key,
  section: "llm",
  section_label: "LLM",
  value_type: valueType,
  default: null,
  choices: [],
  depends_on: [],
  prominence: "critical",
  secret: key === "llm.api_key",
  required: false,
  ...extra,
});

const LLM_SCHEMA = vi.hoisted(() => ({ current: undefined as unknown }));
LLM_SCHEMA.current = {
  model_name: "AgentSettings",
  sections: [
    {
      key: "llm",
      label: "LLM",
      fields: [
        llmField("llm.model"),
        llmField("llm.api_key"),
        llmField("llm.base_url"),
        llmField("llm.temperature", "number"),
        llmField("llm.timeout", "number"),
        llmField("llm.auth_type", "string", {
          default: "api_key",
          choices: [
            { label: "API key", value: "api_key" },
            { label: "Subscription", value: "subscription" },
          ],
        }),
      ],
    },
  ],
};

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { kind: formState.backendKind, id: formState.backendId },
  }),
}));

vi.mock("#/api/profiles-service/profiles-service.api", () => ({
  default: { saveProfile, activateProfile, listProfiles, getProfile },
}));

vi.mock("#/api/settings-service/settings-service.api", () => ({
  default: {
    fetchSettingsFromApi: fetchSettings,
    invalidateCache: vi.fn(),
  },
}));

vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME: "default",
  default: {
    listProfiles: listAgentProfiles,
    saveProfile: applyAgentProfile,
    getProfile: vi
      .fn()
      .mockResolvedValue({ profile: { id: "onboarding-agent" } }),
    activateProfile: vi.fn().mockResolvedValue(undefined),
  },
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({ data: undefined }),
}));

vi.mock("#/hooks/query/use-agent-settings-schema", () => ({
  useAgentSettingsSchema: () => ({ data: LLM_SCHEMA.current, error: null }),
}));

vi.mock("#/hooks/query/use-llm-subscription-models", () => ({
  useOpenAISubscriptionModels: () => ({ data: formState.subscriptionModels }),
}));

vi.mock("#/hooks/query/use-free-models", () => ({
  useDefaultModel: () => null,
  useDefaultModelReady: () => true,
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast,
}));

// Stands in for the embedded LLM form: it starts from the overrides the step
// passes, applies the user's changes from `formState.dirtyLlm` and exposes the
// same save control as the real form.
vi.mock("#/routes/llm-settings", async () => {
  const React = await import("react");
  return {
    LlmSettingsScreen: (props: ScreenProps) => {
      formState.screenProps = props;
      const { onSaveControlChange } = props;
      React.useEffect(() => {
        const values: SettingsFormValues = {
          ...(props.initialValueOverrides ?? {}),
          ...Object.fromEntries(
            Object.entries(formState.dirtyLlm).map(([key, value]) => [
              `llm.${key}`,
              String(value),
            ]),
          ),
        };
        onSaveControlChange({
          save: () => {
            settingsSave();
            props.onSaveSuccess?.();
          },
          isSaving: false,
          isDirty:
            props.markInitialOverridesDirty !== false ||
            Object.keys(formState.dirtyLlm).length > 0,
          values,
          view: formState.view,
          getDirtyPayload: () => ({ llm: formState.dirtyLlm }),
          getSavePayload: () => ({
            agent_settings_diff: { llm: formState.dirtyLlm },
          }),
        });
      }, [onSaveControlChange]);
      return <div data-testid="llm-settings-screen" />;
    },
  };
});

const API_KEY_PROFILE_DEFAULTS = {
  auth_type: "api_key",
  subscription_vendor: null,
  provider_connection_id: null,
};

function givenActiveProfile(name: string, config: Record<string, unknown>) {
  listProfiles.mockResolvedValue({
    profiles: [{ name }],
    active_profile: name,
  });
  getProfile.mockResolvedValue({ name, config, api_key_set: true });
}

// Next stays disabled until the form has mounted from the profile it edits.
async function clickNext() {
  await screen.findByTestId("llm-settings-screen");
  const nextButton = screen.getByTestId("onboarding-llm-next");
  await waitFor(() => expect(nextButton).toBeEnabled());
  await userEvent.click(nextButton);
}

describe("SetupLlmStep", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    formState.backendKind = "local";
    formState.backendId = "local-backend";
    formState.dirtyLlm = {};
    formState.view = "all";
    formState.screenProps = undefined;
    formState.subscriptionModels = undefined;
    listProfiles.mockResolvedValue({ profiles: [], active_profile: null });
    listAgentProfiles.mockResolvedValue({
      profiles: [],
      active_agent_profile_id: null,
    });
    getProfile.mockResolvedValue({ name: "", config: {}, api_key_set: false });
    saveProfile.mockResolvedValue(undefined);
    activateProfile.mockResolvedValue(undefined);
    applyAgentProfile.mockResolvedValue(undefined);
  });

  it("edits the active profile and keeps its unchanged endpoint, key and options", async () => {
    const activeConfig = {
      model: "openai/previous-model",
      base_url: "http://localhost:19118/v1",
      api_key: "encrypted:test-key",
      temperature: 0.2,
      timeout: 30,
    };
    givenActiveProfile("previous", activeConfig);
    formState.dirtyLlm = { model: "openai/mock-onboarding-model" };
    const onNext = vi.fn();
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={onNext} />);

    await clickNext();

    await waitFor(() => expect(onNext).toHaveBeenCalledTimes(1));
    expect(getProfile).toHaveBeenCalledWith("previous", "encrypted");
    // The form shows the active profile, its own model included.
    expect(
      screen.getByText("ONBOARDING$LLM_SUBTITLE_CURRENT_PROFILE"),
    ).toBeInTheDocument();
    expect(formState.screenProps?.markInitialOverridesDirty).toBe(false);
    expect(formState.screenProps?.initialValueOverrides).toMatchObject({
      "llm.model": "openai/previous-model",
      "llm.api_key": "encrypted:test-key",
      "llm.base_url": "http://localhost:19118/v1",
      "llm.temperature": "0.2",
      "llm.timeout": "30",
      "llm.auth_type": "api_key",
    });
    expect(saveProfile).toHaveBeenCalledWith("mock-onboarding-model", {
      llm: {
        ...activeConfig,
        ...API_KEY_PROFILE_DEFAULTS,
        model: "openai/mock-onboarding-model",
      },
      include_secrets: true,
    });
    expect(activateProfile).toHaveBeenCalledWith("mock-onboarding-model");
    expect(applyAgentProfile).toHaveBeenCalledWith("default", {
      agent_kind: "openhands",
      llm_profile_ref: "mock-onboarding-model",
    });
  });

  it("neither reads nor writes the raw LLM settings on a local backend", async () => {
    givenActiveProfile("previous", { model: "openai/previous-model" });
    formState.dirtyLlm = { model: "openai/gpt-4o-mini", api_key: "sk-test" };
    const onNext = vi.fn();
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={onNext} />);

    await clickNext();

    await waitFor(() => expect(onNext).toHaveBeenCalledTimes(1));
    expect(fetchSettings).not.toHaveBeenCalled();
    expect(settingsSave).not.toHaveBeenCalled();
  });

  it("starts from a fresh profile when no profile is active", async () => {
    formState.dirtyLlm = {
      model: "openai/gpt-4o-mini",
      api_key: "sk-test",
      timeout: 3,
    };
    const onNext = vi.fn();
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={onNext} />);

    await clickNext();

    await waitFor(() => expect(onNext).toHaveBeenCalledTimes(1));
    expect(getProfile).not.toHaveBeenCalled();
    expect(screen.getByText("ONBOARDING$LLM_SUBTITLE")).toBeInTheDocument();
    expect(formState.screenProps?.initialValueOverrides).toMatchObject({
      "llm.model": ONBOARDING_DEFAULT_LLM_MODEL,
      "llm.api_key": "",
      "llm.base_url": "",
      "llm.auth_type": "api_key",
    });
    expect(saveProfile).toHaveBeenCalledWith("gpt-4o-mini", {
      llm: {
        ...API_KEY_PROFILE_DEFAULTS,
        model: "openai/gpt-4o-mini",
        api_key: "sk-test",
        timeout: 3,
      },
      include_secrets: true,
    });
  });

  it("starts from a fresh profile when the active profile no longer exists", async () => {
    listProfiles.mockResolvedValue({
      profiles: [{ name: "other" }],
      active_profile: "deleted",
    });
    formState.dirtyLlm = { model: "openai/gpt-4o-mini" };
    const onNext = vi.fn();
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={onNext} />);

    await clickNext();

    await waitFor(() => expect(onNext).toHaveBeenCalledTimes(1));
    expect(getProfile).not.toHaveBeenCalled();
    expect(saveProfile).toHaveBeenCalledWith("gpt-4o-mini", {
      llm: { ...API_KEY_PROFILE_DEFAULTS, model: "openai/gpt-4o-mini" },
      include_secrets: true,
    });
  });

  it("waits for a backend, then fills the form from that backend's active profile", async () => {
    // Public-mode onboarding mounts this step before the backend is added.
    formState.backendId = "no-backend";
    givenActiveProfile("previous", {
      model: "openai/previous-model",
      base_url: "http://localhost:19118/v1",
    });
    const { rerender } = renderWithProviders(
      <SetupLlmStep onBack={vi.fn()} onNext={vi.fn()} />,
    );

    expect(screen.queryByTestId("llm-settings-screen")).toBeNull();
    expect(screen.getByTestId("onboarding-llm-next")).toBeDisabled();
    expect(listProfiles).not.toHaveBeenCalled();
    expect(listAgentProfiles).not.toHaveBeenCalled();

    formState.backendId = "local-backend";
    rerender(<SetupLlmStep onBack={vi.fn()} onNext={vi.fn()} />);

    await screen.findByTestId("llm-settings-screen");
    expect(getProfile).toHaveBeenCalledWith("previous", "encrypted");
    expect(formState.screenProps?.initialValueOverrides).toMatchObject({
      "llm.base_url": "http://localhost:19118/v1",
    });
  });

  it("keeps an unchanged existing profile instead of saving a copy", async () => {
    givenActiveProfile("my-deepseek", {
      model: "deepseek/deepseek-flash",
      base_url: "https://api.deepseek.com/v1",
      api_key: "encrypted:test-key",
    });
    const onNext = vi.fn();
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={onNext} />);

    await clickNext();

    await waitFor(() => expect(onNext).toHaveBeenCalledTimes(1));
    expect(saveProfile).not.toHaveBeenCalled();
    expect(activateProfile).toHaveBeenCalledWith("my-deepseek");
    expect(applyAgentProfile).toHaveBeenCalledWith("default", {
      agent_kind: "openhands",
      llm_profile_ref: "my-deepseek",
    });
  });

  it("saves the prefilled default model for a fresh profile", async () => {
    formState.view = "basic";
    formState.dirtyLlm = { api_key: "sk-test" };
    const onNext = vi.fn();
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={onNext} />);

    await clickNext();

    await waitFor(() => expect(onNext).toHaveBeenCalledTimes(1));
    expect(saveProfile).toHaveBeenCalledWith("gpt-5.6-sol", {
      llm: {
        ...API_KEY_PROFILE_DEFAULTS,
        model: ONBOARDING_DEFAULT_LLM_MODEL,
        api_key: "sk-test",
      },
      include_secrets: true,
    });
  });

  it("drops the hidden Base URL when the model changes in the Basic view", async () => {
    givenActiveProfile("previous", {
      model: "deepseek/deepseek-flash",
      base_url: "https://api.deepseek.com/v1",
      api_key: "encrypted:test-key",
    });
    formState.view = "basic";
    formState.dirtyLlm = { model: "openai/gpt-4o-mini" };
    const onNext = vi.fn();
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={onNext} />);

    await clickNext();

    await waitFor(() => expect(onNext).toHaveBeenCalledTimes(1));
    expect(saveProfile).toHaveBeenCalledWith("gpt-4o-mini", {
      llm: {
        ...API_KEY_PROFILE_DEFAULTS,
        model: "openai/gpt-4o-mini",
        api_key: "encrypted:test-key",
      },
      include_secrets: true,
    });
  });

  it("falls back to the profile the default agent profile points at", async () => {
    // An older install: the Agent Server's backfill turned its raw LLM
    // settings into the `default` LLM profile without activating it.
    listProfiles.mockResolvedValue({
      profiles: [{ name: "default" }],
      active_profile: null,
    });
    listAgentProfiles.mockResolvedValue({
      profiles: [{ name: "default", llm_profile_ref: "default" }],
      active_agent_profile_id: "agent-1",
    });
    getProfile.mockResolvedValue({
      name: "default",
      config: {
        model: "openai/legacy-model",
        base_url: "http://legacy.example/v1",
        api_key: "encrypted:legacy-key",
      },
      api_key_set: true,
    });
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={vi.fn()} />);

    await screen.findByTestId("llm-settings-screen");
    expect(getProfile).toHaveBeenCalledWith("default", "encrypted");
    // Listing the agent profiles is what runs the backfill, so it comes first.
    expect(listAgentProfiles.mock.invocationCallOrder[0]).toBeLessThan(
      listProfiles.mock.invocationCallOrder[0],
    );
    expect(formState.screenProps?.initialValueOverrides).toMatchObject({
      "llm.model": "openai/legacy-model",
      "llm.base_url": "http://legacy.example/v1",
      "llm.api_key": "encrypted:legacy-key",
    });
  });

  it("prefers the active LLM profile over the default agent profile's", async () => {
    listProfiles.mockResolvedValue({
      profiles: [{ name: "active-one" }, { name: "default" }],
      active_profile: "active-one",
    });
    listAgentProfiles.mockResolvedValue({
      profiles: [{ name: "default", llm_profile_ref: "default" }],
      active_agent_profile_id: "agent-1",
    });
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={vi.fn()} />);

    await screen.findByTestId("llm-settings-screen");
    expect(getProfile).toHaveBeenCalledTimes(1);
    expect(getProfile).toHaveBeenCalledWith("active-one", "encrypted");
  });

  it("shows an error with a retry instead of a fresh form when the profiles can't be read", async () => {
    listProfiles.mockRejectedValue(new Error("backend unavailable"));
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={vi.fn()} />);

    await screen.findByTestId(
      "onboarding-llm-load-error",
      {},
      { timeout: 4000 },
    );
    expect(screen.queryByTestId("llm-settings-screen")).toBeNull();
    expect(screen.getByTestId("onboarding-llm-next")).toBeDisabled();

    givenActiveProfile("previous", { model: "openai/previous-model" });
    await userEvent.click(screen.getByTestId("onboarding-llm-load-retry"));

    await screen.findByTestId("llm-settings-screen");
    expect(formState.screenProps?.initialValueOverrides).toMatchObject({
      "llm.model": "openai/previous-model",
    });
  });

  it("saves a subscription profile without an inline key or Base URL", async () => {
    givenActiveProfile("gpt-5.6-luna", {
      model: "gpt-5.6-luna",
      auth_type: "subscription",
      subscription_vendor: "openai",
      temperature: 0.2,
    });
    formState.dirtyLlm = { model: "gpt-5.6-luna" };
    formState.subscriptionModels = ["gpt-5.6-luna", "gpt-5.6-sol"];
    const onNext = vi.fn();
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={onNext} />);

    await clickNext();

    await waitFor(() => expect(onNext).toHaveBeenCalledTimes(1));
    expect(saveProfile).toHaveBeenCalledWith("gpt-5.6-luna", {
      llm: {
        model: "gpt-5.6-luna",
        auth_type: "subscription",
        subscription_vendor: "openai",
        provider_connection_id: null,
        temperature: 0.2,
      },
      include_secrets: true,
    });
    expect(activateProfile).toHaveBeenCalledWith("gpt-5.6-luna");
  });

  it("saves the subscription model the picker shows when the form still holds another", async () => {
    // Switching to subscription before its models load leaves the form's
    // value at the default model while the picker shows the first model.
    formState.dirtyLlm = { auth_type: "subscription" };
    formState.subscriptionModels = ["gpt-5.6-luna", "gpt-5.6-sol-codex"];
    const onNext = vi.fn();
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={onNext} />);

    await clickNext();

    await waitFor(() => expect(onNext).toHaveBeenCalledTimes(1));
    expect(saveProfile).toHaveBeenCalledWith("gpt-5.6-luna", {
      llm: {
        model: "gpt-5.6-luna",
        auth_type: "subscription",
        subscription_vendor: "openai",
        provider_connection_id: null,
      },
      include_secrets: true,
    });
  });

  it("waits for the subscription models before saving a subscription profile", async () => {
    formState.dirtyLlm = { auth_type: "subscription" };
    const onNext = vi.fn();
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={onNext} />);

    await clickNext();

    await waitFor(() =>
      expect(displayErrorToast).toHaveBeenCalledWith(
        "Subscription models are not loaded yet.",
      ),
    );
    expect(saveProfile).not.toHaveBeenCalled();
    expect(onNext).not.toHaveBeenCalled();
  });

  it("keeps Cloud onboarding on its settings save path", async () => {
    formState.backendKind = "cloud";
    const onNext = vi.fn();
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={onNext} />);

    await clickNext();

    await waitFor(() => expect(onNext).toHaveBeenCalledTimes(1));
    expect(settingsSave).toHaveBeenCalledTimes(1);
    expect(formState.screenProps?.initialValueOverrides).toEqual({
      "llm.model": ONBOARDING_DEFAULT_LLM_MODEL,
    });
    expect(formState.screenProps?.markInitialOverridesDirty).toBe(true);
    expect(listProfiles).not.toHaveBeenCalled();
    expect(listAgentProfiles).not.toHaveBeenCalled();
    expect(getProfile).not.toHaveBeenCalled();
    expect(saveProfile).not.toHaveBeenCalled();
    expect(activateProfile).not.toHaveBeenCalled();
    expect(applyAgentProfile).not.toHaveBeenCalled();
  });

  it.each([
    ["profile save", saveProfile],
    ["profile activation", activateProfile],
  ])("does not advance when the %s fails", async (_label, failingCall) => {
    failingCall.mockRejectedValueOnce(new Error("profile unavailable"));
    formState.dirtyLlm = { model: "openai/gpt-4o-mini", api_key: "sk-test" };
    const onNext = vi.fn();
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={onNext} />);

    await clickNext();

    await waitFor(() => expect(displayErrorToast).toHaveBeenCalled());
    expect(onNext).not.toHaveBeenCalled();
    const nextButton = screen.getByTestId("onboarding-llm-next");
    await waitFor(() => expect(nextButton).toBeEnabled());

    await userEvent.click(nextButton);
    await waitFor(() => expect(onNext).toHaveBeenCalledTimes(1));
    expect(applyAgentProfile).toHaveBeenLastCalledWith("default", {
      agent_kind: "openhands",
      llm_profile_ref: "gpt-4o-mini",
    });
  });
});
