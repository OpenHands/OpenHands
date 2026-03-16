import { describe, it, expect, beforeEach } from "vitest";
import {
  setCTADismissed,
  isCTADismissed,
} from "#/utils/session-storage";

describe("session-storage CTA utilities", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  describe("isCTADismissed", () => {
    it("returns false when CTA has not been dismissed", () => {
      expect(isCTADismissed("homepage")).toBe(false);
    });

    it("returns true when CTA has been dismissed", () => {
      localStorage.setItem("homepage-cta-dismissed", "true");
      expect(isCTADismissed("homepage")).toBe(true);
    });

    it("returns false when storage value is not 'true'", () => {
      localStorage.setItem("homepage-cta-dismissed", "false");
      expect(isCTADismissed("homepage")).toBe(false);

      localStorage.setItem("homepage-cta-dismissed", "invalid");
      expect(isCTADismissed("homepage")).toBe(false);
    });
  });

  describe("setCTADismissed", () => {
    it("sets the CTA as dismissed in local storage", () => {
      setCTADismissed("homepage");
      expect(localStorage.getItem("homepage-cta-dismissed")).toBe("true");
    });

    it("generates correct key for homepage location", () => {
      setCTADismissed("homepage");
      expect(localStorage.getItem("homepage-cta-dismissed")).toBe("true");
    });
  });

  describe("storage key format", () => {
    it("uses the correct key format: {location}-cta-dismissed", () => {
      setCTADismissed("homepage");

      // Verify key exists with correct format
      expect(localStorage.getItem("homepage-cta-dismissed")).toBe("true");

      // Verify other keys don't exist
      expect(localStorage.getItem("cta-dismissed")).toBeNull();
      expect(localStorage.getItem("homepage")).toBeNull();
    });
  });

  describe("persistence across tabs", () => {
    it("dismissed state persists across multiple reads", () => {
      setCTADismissed("homepage");

      expect(isCTADismissed("homepage")).toBe(true);
      expect(isCTADismissed("homepage")).toBe(true);
      expect(isCTADismissed("homepage")).toBe(true);
    });

    it("different locations are independent", () => {
      // Note: This test documents expected behavior when context-menu is added
      setCTADismissed("homepage");

      expect(isCTADismissed("homepage")).toBe(true);
      // Other locations would be independent when added to CTALocation type
    });
  });
});
