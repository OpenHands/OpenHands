import { render, screen } from "@testing-library/react";
import { it, describe, expect } from "vitest";
import { TermsAndPrivacyNotice } from "#/components/shared/terms-and-privacy-notice";

describe("TermsAndPrivacyNotice", () => {
  it("should render Terms of Service and Privacy Policy links", () => {
    // Arrange & Act
    render(<TermsAndPrivacyNotice />);

    // Assert
    const termsSection = screen.getByTestId("terms-and-privacy-notice");
    expect(termsSection).toBeInTheDocument();

    const tosLink = screen.getByRole("link", {
      name: "COMMON$TERMS_OF_SERVICE",
    });
    const privacyLink = screen.getByRole("link", {
      name: "COMMON$PRIVACY_POLICY",
    });

    expect(tosLink).toBeInTheDocument();
    expect(tosLink).toHaveAttribute("href", "https://www.all-hands.dev/tos");
    expect(tosLink).toHaveAttribute("target", "_blank");
    expect(tosLink).toHaveAttribute("rel", "noopener noreferrer");

    expect(privacyLink).toBeInTheDocument();
    expect(privacyLink).toHaveAttribute(
      "href",
      "https://www.all-hands.dev/privacy",
    );
    expect(privacyLink).toHaveAttribute("target", "_blank");
    expect(privacyLink).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("should render all required text content", () => {
    // Arrange & Act
    render(<TermsAndPrivacyNotice />);

    // Assert
    const termsSection = screen.getByTestId("terms-and-privacy-notice");
    expect(termsSection).toHaveTextContent(
      "AUTH$BY_SIGNING_UP_YOU_AGREE_TO_OUR",
    );
    expect(termsSection).toHaveTextContent("COMMON$TERMS_OF_SERVICE");
    expect(termsSection).toHaveTextContent("COMMON$AND");
    expect(termsSection).toHaveTextContent("COMMON$PRIVACY_POLICY");
  });
});
