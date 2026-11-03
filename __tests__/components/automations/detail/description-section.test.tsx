import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { I18nKey } from "#/i18n/declaration";
import { DescriptionSection } from "#/components/features/automations/detail/description-section";

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
  });
});
