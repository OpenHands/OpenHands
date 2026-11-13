import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { I18nKey } from "#/i18n/declaration";
import { DescriptionSection } from "#/components/features/automations/detail/description-section";

vi.mock("mermaid", () => ({
  default: {
    initialize: vi.fn(),
    parse: vi.fn().mockResolvedValue({ diagramType: "flowchart" }),
    render: vi.fn().mockResolvedValue({
      svg: '<svg data-testid="template-diagram-svg"></svg>',
    }),
  },
}));

const TEMPLATE_DIAGRAM = "flowchart TD\n  T --> U";

describe("DescriptionSection", () => {
  it("renders the description as markdown under its title", () => {
    render(
      <DescriptionSection
        description={"Reviews **pull requests**.\n\n- step one"}
      />,
    );

    expect(
      screen.getByText(I18nKey.AUTOMATIONS$DETAIL$DESCRIPTION),
    ).toBeInTheDocument();
    const content = screen.getByTestId("automation-description-content");
    expect(content.querySelector("strong")).toHaveTextContent("pull requests");
    expect(screen.getByText("step one").tagName).toBe("LI");
    expect(
      screen.queryByTestId("mermaid-diagram-pending"),
    ).not.toBeInTheDocument();
  });

  it("draws a given diagram under the description", async () => {
    // Act
    render(
      <DescriptionSection
        description="Reviews pull requests."
        diagram={TEMPLATE_DIAGRAM}
      />,
    );

    // Assert
    const content = screen.getByTestId("automation-description-content");
    expect(content).toHaveTextContent("Reviews pull requests.");
    expect(await screen.findByTestId("mermaid-diagram")).toContainElement(
      screen.getByTestId("template-diagram-svg"),
    );
  });
});
