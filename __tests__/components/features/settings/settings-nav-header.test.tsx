import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { SettingsNavHeader } from "#/components/features/settings/settings-nav-header";
import { I18nKey } from "#/i18n/declaration";

describe("SettingsNavHeader", () => {
  it("should render the translated header text", () => {
    render(<SettingsNavHeader text={I18nKey.SETTINGS$TITLE} />);

    expect(
      screen.getByText("SETTINGS$TITLE"),
    ).toBeInTheDocument();
  });

  // Prop-passthrough contract: consumer-supplied className must land on the
  // element.
  it("should accept custom className", () => {
    const { container } = render(
      <SettingsNavHeader
        text={I18nKey.SETTINGS$TITLE}
        className="custom-class"
      />,
    );

    const wrapper = container.firstChild;
    expect(wrapper).toHaveClass("custom-class");
  });
});
