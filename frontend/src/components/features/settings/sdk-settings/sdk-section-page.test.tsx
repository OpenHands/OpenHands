import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { usePermission } from "#/hooks/organizations/use-permissions";
import {
  useAgentSettingsSchema,
  useConversationSettingsSchema,
} from "#/hooks/query/use-agent-settings-schema";
import { useConfig } from "#/hooks/query/use-config";
import { useMe } from "#/hooks/query/use-me";
import { useSettings } from "#/hooks/query/use-settings";
import { DEFAULT_SETTINGS } from "#/services/settings";
import type { Settings, SettingsSchema } from "#/types/settings";
import { SdkSectionPage, type SdkSectionHeaderProps } from "./sdk-section-page";

vi.mock("#/hooks/mutation/use-save-settings", () => ({
  useSaveSettings: vi.fn(),
}));
vi.mock("#/hooks/organizations/use-permissions", () => ({
  usePermission: vi.fn(),
}));
vi.mock("#/hooks/query/use-agent-settings-schema", () => ({
  useAgentSettingsSchema: vi.fn(),
  useConversationSettingsSchema: vi.fn(),
}));
vi.mock("#/hooks/query/use-config", () => ({
  useConfig: vi.fn(),
}));
vi.mock("#/hooks/query/use-me", () => ({
  useMe: vi.fn(),
}));
vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: vi.fn(),
}));

const mockedUseSaveSettings = vi.mocked(useSaveSettings);
const mockedUsePermission = vi.mocked(usePermission);
const mockedUseAgentSettingsSchema = vi.mocked(useAgentSettingsSchema);
const mockedUseConversationSettingsSchema = vi.mocked(
  useConversationSettingsSchema,
);
const mockedUseConfig = vi.mocked(useConfig);
const mockedUseMe = vi.mocked(useMe);
const mockedUseSettings = vi.mocked(useSettings);

const mutateSpy = vi.fn();

const createBooleanSchema = (
  sectionKey: string,
  fieldKey: string,
  defaultValue: boolean,
): SettingsSchema => ({
  model_name: sectionKey,
  sections: [
    {
      key: sectionKey,
      label: sectionKey,
      fields: [
        {
          key: fieldKey,
          label: fieldKey,
          description: null,
          section: sectionKey,
          section_label: sectionKey,
          value_type: "boolean",
          default: defaultValue,
          choices: [],
          depends_on: [],
          prominence: "critical",
          secret: false,
          required: false,
        },
      ],
    },
  ],
});

const agentSchema = createBooleanSchema("condenser", "condenser.enabled", true);
const conversationSchema = createBooleanSchema(
  "verification",
  "confirmation_mode",
  false,
);

const makeSettings = (overrides: Partial<Settings> = {}): Settings => ({
  ...DEFAULT_SETTINGS,
  agent_settings_schema: agentSchema,
  conversation_settings_schema: conversationSchema,
  agent_settings: { condenser: { enabled: true } },
  conversation_settings: { confirmation_mode: false },
  ...overrides,
});

const makeHeader = (
  testId: string,
  fieldKey: string,
  nextValue: string | boolean,
) =>
  function TestHeader({ onChange }: SdkSectionHeaderProps) {
    return (
      <button
        type="button"
        data-testid={testId}
        onClick={() => onChange(fieldKey, nextValue)}
      >
        {testId}
      </button>
    );
  };

describe("SdkSectionPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();

    mockedUseSaveSettings.mockReturnValue({
      mutate: mutateSpy,
      isPending: false,
    } as never);
    mockedUsePermission.mockReturnValue({
      hasPermission: () => true,
    } as never);
    mockedUseConfig.mockReturnValue({
      data: { app_mode: "oss" },
    } as never);
    mockedUseMe.mockReturnValue({
      data: { role: "owner" },
    } as never);
    mockedUseAgentSettingsSchema.mockReturnValue({
      data: agentSchema,
      isLoading: false,
      isFetching: false,
    } as never);
    mockedUseConversationSettingsSchema.mockReturnValue({
      data: conversationSchema,
      isLoading: false,
      isFetching: false,
    } as never);
  });

  it("posts agent_settings_diff by default for org-scoped agent settings sections", async () => {
    const user = userEvent.setup();
    mockedUseSettings.mockReturnValue({
      data: makeSettings(),
      isLoading: false,
      isFetching: false,
    } as never);

    render(
      <SdkSectionPage
        scope="org"
        sectionKeys={["condenser"]}
        header={makeHeader("change-agent-setting", "condenser.enabled", false)}
      />,
    );

    await user.click(screen.getByTestId("change-agent-setting"));
    await user.click(screen.getByTestId("save-button"));

    expect(mutateSpy).toHaveBeenCalledWith(
      { agent_settings_diff: { condenser: { enabled: false } } },
      expect.any(Object),
    );
  });

  it("posts conversation_settings_diff by default for org-scoped conversation settings sections", async () => {
    const user = userEvent.setup();
    mockedUseSettings.mockReturnValue({
      data: makeSettings(),
      isLoading: false,
      isFetching: false,
    } as never);

    render(
      <SdkSectionPage
        scope="org"
        settingsSource="conversation_settings"
        sectionKeys={["verification"]}
        header={makeHeader(
          "change-conversation-setting",
          "confirmation_mode",
          true,
        )}
      />,
    );

    await user.click(screen.getByTestId("change-conversation-setting"));
    await user.click(screen.getByTestId("save-button"));

    expect(mutateSpy).toHaveBeenCalledWith(
      { conversation_settings_diff: { confirmation_mode: true } },
      expect.any(Object),
    );
  });
});
