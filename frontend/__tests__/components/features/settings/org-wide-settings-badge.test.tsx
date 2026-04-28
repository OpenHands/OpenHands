import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { OrgWideSettingsBadge } from "#/components/features/settings/org-wide-settings-badge";

describe("OrgWideSettingsBadge", () => {
  it("should render the badge with translated text", () => {
    // Arrange & Act
    render(<OrgWideSettingsBadge />);

    // Assert
    const badge = screen.getByTestId("org-wide-settings-badge");
    expect(badge).toBeInTheDocument();
    expect(screen.getByText("SETTINGS$ORG_WIDE_SETTING_BADGE")).toBeInTheDocument();
  });

  it("should render the info circle icon", () => {
    // Arrange & Act
    render(<OrgWideSettingsBadge />);

    // Assert
    const badge = screen.getByTestId("org-wide-settings-badge");
    const icon = badge.querySelector("svg");
    expect(icon).toBeInTheDocument();
  });

  it("should render the managed-by-admin i18n key when variant is set", () => {
    // Arrange & Act
    render(<OrgWideSettingsBadge variant="managed-by-admin" />);

    // Assert
    expect(
      screen.getByText("SETTINGS$ORG_MANAGED_BY_ADMIN_BADGE"),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("SETTINGS$ORG_WIDE_SETTING_BADGE"),
    ).not.toBeInTheDocument();
  });
});
