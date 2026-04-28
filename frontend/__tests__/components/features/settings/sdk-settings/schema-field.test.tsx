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
});
