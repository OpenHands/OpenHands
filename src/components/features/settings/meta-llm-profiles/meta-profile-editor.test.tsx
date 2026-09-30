import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";
import type { MetaProfile } from "#/api/meta-profiles-service/meta-profiles-service.api";
import { MetaProfileEditor } from "./meta-profile-editor";
import {
  DEFAULT_ROUTER_PRO_META_PROFILE_DEFAULT,
  DEFAULT_ROUTER_PRO_META_PROFILE_NAME,
} from "./default-meta-profile";

const connection: ProviderConnection = {
  id: "conn-1",
  display_name: "My OpenAI",
  provider: "openai",
  base_url: null,
  created_at: 1,
  updated_at: 2,
  api_key_set: true,
};

vi.mock("#/components/features/settings/settings-dropdown-input", () => ({
  SettingsDropdownInput: ({
    testId,
    items,
    defaultSelectedKey,
    onSelectionChange,
    onInputChange,
  }: {
    testId: string;
    items: { key: string; label: string }[];
    defaultSelectedKey?: string;
    onSelectionChange?: (key: string | null) => void;
    onInputChange?: (value: string) => void;
  }) => (
    <select
      data-testid={testId}
      value={defaultSelectedKey ?? ""}
      onChange={(event) => {
        onSelectionChange?.(event.target.value || null);
        onInputChange?.(event.target.value);
      }}
    >
      {items.map((item) => (
        <option key={item.key} value={item.key}>
          {item.label}
        </option>
      ))}
    </select>
  ),
}));

function renderEditor(
  overrides: {
    availableProfiles?: string[];
    initialConfig?: MetaProfile;
  } = {},
) {
  const defaultConfig = JSON.parse(
    JSON.stringify(DEFAULT_ROUTER_PRO_META_PROFILE_DEFAULT),
  ) as MetaProfile;
  const config = overrides.initialConfig ?? defaultConfig;
  const onSave = vi.fn();

  renderWithProviders(
    <MetaProfileEditor
      mode="create"
      initialName={DEFAULT_ROUTER_PRO_META_PROFILE_NAME}
      initialConfig={config}
      selectRouterConnectionByDefault
      providerConnections={[connection]}
      availableProfiles={overrides.availableProfiles ?? []}
      isSaving={false}
      onSave={onSave}
      onCancel={vi.fn()}
    />,
  );
  return { onSave };
}

describe("MetaProfileEditor", () => {
  it("preselects the template classifier when no LLM profiles exist yet", () => {
    renderEditor({ availableProfiles: [] });

    const classifierInput = screen.getByTestId(
      "meta-profile-classifier-input",
    ) as HTMLSelectElement;
    expect(classifierInput.options).toHaveLength(1);
    expect(classifierInput.options[0].value).toBe("minimax-m3");
    expect(classifierInput.value).toBe("minimax-m3");
  });

  it("keeps an orphaned classifier value as an option instead of dropping it", () => {
    const config: MetaProfile = {
      classifier_model: "custom-classifier",
      classes: [],
      prompt_template:
        DEFAULT_ROUTER_PRO_META_PROFILE_DEFAULT.prompt_template ?? "",
      model_table: DEFAULT_ROUTER_PRO_META_PROFILE_DEFAULT.model_table,
    };

    renderEditor({
      initialConfig: config,
      availableProfiles: ["gpt-5.4"],
    });

    const classifierInput = screen.getByTestId(
      "meta-profile-classifier-input",
    ) as HTMLSelectElement;
    expect(classifierInput.options[0].value).toBe("gpt-5.4");
    expect(classifierInput.options[1].value).toBe("custom-classifier");
    expect(classifierInput.value).toBe("custom-classifier");
  });

  it("appends the classifier to the save payload when creating from the template with no profiles", () => {
    const { onSave } = renderEditor({ availableProfiles: [] });

    screen.getByTestId("meta-profile-save").click();

    expect(onSave).toHaveBeenCalledWith(
      DEFAULT_ROUTER_PRO_META_PROFILE_NAME,
      expect.objectContaining({
        classifier_model: "minimax-m3",
        model_table: DEFAULT_ROUTER_PRO_META_PROFILE_DEFAULT.model_table,
      }),
      "conn-1",
    );
  });
});
