import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { SettingsNavHeader } from "#/components/features/settings/settings-nav-header";
import { I18nKey } from "#/i18n/declaration";

describe("SettingsNavHeader", () => {
  it("should render the translated header text", () => {
    // Arrange & Act
    render(<SettingsNavHeader text={I18nKey.SETTINGS$ORG_SETTINGS_HEADER} />);

    // Assert
    expect(screen.getByText("SETTINGS$ORG_SETTINGS_HEADER")).toBeInTheDocument();
  });

  it("should render different header text based on prop", () => {
    // Arrange & Act
    render(<SettingsNavHeader text={I18nKey.SETTINGS$PERSONAL_SETTINGS_HEADER} />);

    // Assert
    expect(screen.getByText("SETTINGS$PERSONAL_SETTINGS_HEADER")).toBeInTheDocument();
  });

  it("should accept custom className", () => {
    // Arrange & Act
    const { container } = render(
      <SettingsNavHeader
        text={I18nKey.SETTINGS$ORG_SETTINGS_HEADER}
        className="px-2 pt-2 pb-1"
      />,
    );

    // Assert
    const wrapper = container.firstChild;
    expect(wrapper).toHaveClass("px-2");
    expect(wrapper).toHaveClass("pt-2");
    expect(wrapper).toHaveClass("pb-1");
  });
});
