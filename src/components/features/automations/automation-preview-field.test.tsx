import type { ComponentProps } from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AutomationPreviewField } from "./automation-preview-field";

function renderRow(props: ComponentProps<typeof AutomationPreviewField>) {
  const { container } = render(
    <dl>
      <AutomationPreviewField {...props} />
    </dl>,
  );
  return container.querySelector("dl > div")!;
}

describe("AutomationPreviewField", () => {
  it("keeps a short value on the label's row", () => {
    const row = renderRow({ label: "Timezone", value: "Europe/Zurich" });

    expect(row.className).toContain("justify-between");
    expect(row.className).not.toContain("flex-col");
  });

  it("stacks a value too long to sit beside its label", () => {
    const row = renderRow({ label: "Prompt", value: "x".repeat(49) });

    expect(row.className).toContain("flex-col");
  });

  it("stacks a multi-line value", () => {
    const row = renderRow({ label: "Feeds", value: "first\nsecond" });

    expect(row.className).toContain("flex-col");
  });

  it("honours an explicitly stacked layout for short copy", () => {
    const row = renderRow({
      label: "Prompt",
      value: "Short",
      layout: "stacked",
    });

    expect(row.className).toContain("flex-col");
  });

  it("keeps a lone short chip beside its label", () => {
    const row = renderRow({ label: "Plugins", value: "", chips: ["github"] });

    expect(row.className).toContain("justify-between");
  });

  it("stacks once a chip is longer than the row can hold", () => {
    const row = renderRow({
      label: "Feeds",
      value: "",
      chips: ["y".repeat(60)],
    });

    expect(row.className).toContain("flex-col");
  });

  it("renders every chip as its own pill", () => {
    renderRow({ label: "Plugins", value: "", chips: ["one", "two"] });

    expect(screen.getByText("one")).toBeInTheDocument();
    expect(screen.getByText("two")).toBeInTheDocument();
  });
});
