import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { SettingsNavDivider } from "#/components/features/settings/settings-nav-divider";

describe("SettingsNavDivider", () => {
  it("should render the divider element", () => {
    // Arrange & Act
    const { container } = render(<SettingsNavDivider />);

    // Assert
    const divider = container.firstChild;
    expect(divider).toBeInTheDocument();
  });

  // Prop-passthrough contract: consumer-supplied className must land on the
  // element.
  it("should accept custom className", () => {
    // Arrange & Act
    const { container } = render(<SettingsNavDivider className="custom-class" />);

    // Assert
    const divider = container.firstChild;
    expect(divider).toHaveClass("custom-class");
  });
});
