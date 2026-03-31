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

  it("should accept custom className", () => {
    // Arrange & Act
    const { container } = render(<SettingsNavDivider className="my-4" />);

    // Assert
    const divider = container.firstChild;
    expect(divider).toHaveClass("my-4");
  });
});
