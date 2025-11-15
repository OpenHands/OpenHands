import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router";
import i18n from "../../src/i18n";
import { AccountSettingsContextMenu } from "../../src/components/features/context-menu/account-settings-context-menu";
import { renderWithProviders } from "../../test-utils";

describe("Translations", () => {
  it("should render translated text", () => {
    i18n.changeLanguage("en");
    renderWithProviders(
      <MemoryRouter>
        <AccountSettingsContextMenu onLogout={() => {}} onClose={() => {}} />
      </MemoryRouter>,
    );
    expect(
      screen.getByTestId("account-settings-context-menu"),
    ).toBeInTheDocument();
  });

  it("should not attempt to load unsupported language codes", async () => {
    // Test that the configuration prevents 404 errors by not attempting to load
    // unsupported language codes like 'en-US@posix'
    const originalLanguage = i18n.language;

    try {
      // With nonExplicitSupportedLngs: false, i18next will not attempt to load
      // unsupported language codes, preventing 404 errors

      // Test with a language code that includes region but is not in supportedLngs
      await i18n.changeLanguage("en-US@posix");

      // Since "en-US@posix" is not in supportedLngs and nonExplicitSupportedLngs is false,
      // i18next should fall back to the fallbackLng ("en")
      expect(i18n.language).toBe("en");

      // Test another unsupported region code
      await i18n.changeLanguage("ja-JP");

      // Even with nonExplicitSupportedLngs: false, i18next still falls back to base language
      // if it exists in supportedLngs, but importantly, it won't make a 404 request first
      expect(i18n.language).toBe("ja");

      // Test that supported languages still work
      await i18n.changeLanguage("ja");
      expect(i18n.language).toBe("ja");

      await i18n.changeLanguage("zh-CN");
      expect(i18n.language).toBe("zh-CN");
    } finally {
      // Restore the original language
      await i18n.changeLanguage(originalLanguage);
    }
  });

  it("should have proper i18n configuration", () => {
    // Test that the i18n instance has the expected configuration
    expect(i18n.options.supportedLngs).toBeDefined();

    // nonExplicitSupportedLngs should be false to prevent 404 errors
    expect(i18n.options.nonExplicitSupportedLngs).toBe(false);

    // fallbackLng can be a string or array, check if it includes "en"
    const { fallbackLng } = i18n.options;
    if (Array.isArray(fallbackLng)) {
      expect(fallbackLng).toContain("en");
    } else {
      expect(fallbackLng).toBe("en");
    }

    // Test that supported languages include both base and region-specific codes
    const supportedLngs = i18n.options.supportedLngs as string[];
    expect(supportedLngs).toContain("en");
    expect(supportedLngs).toContain("zh-CN");
    expect(supportedLngs).toContain("zh-TW");
    expect(supportedLngs).toContain("ko-KR");
  });
});
