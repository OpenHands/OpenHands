import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SchemaField } from "#/components/features/settings/sdk-settings/schema-field";
import { SettingsFieldSchema } from "#/types/settings";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) =>
      ({
        SETTINGS$TOP_P_LABEL: "Top P",
        SETTINGS$TOP_P_DESCRIPTION: "Controls nucleus sampling.",
        SCHEMA$VERIFICATION$CRITIC_API_KEY$HELP_TEXT:
          "If OpenHands is selected as your active LLM provider, leave this empty because the Critic API Key is the same as your OpenHands Provider LLM Key, which you can find in the",
        SCHEMA$VERIFICATION$CRITIC_API_KEY$HELP_SUFFIX:
          "tab of OpenHands Cloud; otherwise, enter a Critic API Key from that page.",
        SETTINGS$NAV_API_KEYS: "API Keys",
      })[key] ?? key,
  }),
}));

function buildField(
  overrides: Partial<SettingsFieldSchema> = {},
): SettingsFieldSchema {
  return {
    key: "llm.top_p",
    label: "Top P",
    description: "Controls nucleus sampling.",
    section: "llm",
    section_label: "LLM",
    value_type: "number",
    default: 1,
    choices: [],
    depends_on: [],
    prominence: "major",
    secret: false,
    required: false,
    ...overrides,
  };
}

describe("SchemaField", () => {
  it("constrains the Top P input to the valid numeric range", () => {
    render(
      <SchemaField
        field={buildField()}
        value="1"
        isDisabled={false}
        onChange={() => {}}
      />,
    );

    const input = screen.getByTestId("sdk-settings-llm.top_p");

    expect(input).toHaveAttribute("min", "0");
    expect(input).toHaveAttribute("max", "1");
    expect(input).toHaveAttribute("step", "0.01");
  });

  it("translates schema-backed labels and descriptions", () => {
    render(
      <SchemaField
        field={buildField({
          label: "SETTINGS$TOP_P_LABEL",
          description: "SETTINGS$TOP_P_DESCRIPTION",
        })}
        value="1"
        isDisabled={false}
        onChange={() => {}}
      />,
    );

    expect(screen.getByText("Top P")).toBeInTheDocument();
    expect(screen.getByText("Controls nucleus sampling.")).toBeInTheDocument();
  });

  it("renders critic API key guidance as one settings-sized help line", () => {
    render(
      <SchemaField
        field={buildField({
          key: "verification.critic_api_key",
          label: "Critic API Key",
          description: "Server schema description should be replaced.",
          value_type: "string",
          secret: true,
        })}
        value=""
        isDisabled={false}
        onChange={() => {}}
      />,
    );

    const help = screen.getByTestId("help-link-verification.critic_api_key");

    expect(help).toHaveTextContent(
      "Critic API Key is the same as your OpenHands Provider LLM Key",
    );
    expect(help).toHaveTextContent("API Keys");
    expect(help).toHaveClass("text-sm");
    expect(help).toHaveClass("font-normal");
    expect(
      screen.queryByText("Server schema description should be replaced."),
    ).not.toBeInTheDocument();
  });
});
