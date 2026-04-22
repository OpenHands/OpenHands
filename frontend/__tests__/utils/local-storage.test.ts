import { describe, it, expect, beforeEach } from "vitest";
import {
  LOCAL_STORAGE_KEYS,
  LoginMethod,
  setLoginMethod,
  getLoginMethod,
  clearLoginData,
  setCTADismissed,
  isCTADismissed,
} from "#/utils/local-storage";

describe("local-storage utilities", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  describe("Login method utilities", () => {
    describe("setLoginMethod", () => {
      it("stores the login method in local storage", () => {
        setLoginMethod(LoginMethod.GITHUB);
        expect(localStorage.getItem(LOCAL_STORAGE_KEYS.LOGIN_METHOD)).toBe("github");
      });

      it("stores different login methods correctly", () => {
        setLoginMethod(LoginMethod.GITLAB);
        expect(localStorage.getItem(LOCAL_STORAGE_KEYS.LOGIN_METHOD)).toBe("gitlab");

        setLoginMethod(LoginMethod.BITBUCKET);
        expect(localStorage.getItem(LOCAL_STORAGE_KEYS.LOGIN_METHOD)).toBe("bitbucket");

        setLoginMethod(LoginMethod.AZURE_DEVOPS);
        expect(localStorage.getItem(LOCAL_STORAGE_KEYS.LOGIN_METHOD)).toBe("azure_devops");

        setLoginMethod(LoginMethod.ENTERPRISE_SSO);
        expect(localStorage.getItem(LOCAL_STORAGE_KEYS.LOGIN_METHOD)).toBe("enterprise_sso");

        setLoginMethod(LoginMethod.BITBUCKET_DATA_CENTER);
        expect(localStorage.getItem(LOCAL_STORAGE_KEYS.LOGIN_METHOD)).toBe("bitbucket_data_center");
      });

      it("overwrites previous login method", () => {
        setLoginMethod(LoginMethod.GITHUB);
        setLoginMethod(LoginMethod.GITLAB);
        expect(localStorage.getItem(LOCAL_STORAGE_KEYS.LOGIN_METHOD)).toBe("gitlab");
      });
    });

    describe("getLoginMethod", () => {
      it("returns null when no login method is set", () => {
        expect(getLoginMethod()).toBeNull();
      });

      it("returns the stored login method", () => {
        localStorage.setItem(LOCAL_STORAGE_KEYS.LOGIN_METHOD, "github");
        expect(getLoginMethod()).toBe(LoginMethod.GITHUB);
      });

      it("returns correct login method for all types", () => {
        localStorage.setItem(LOCAL_STORAGE_KEYS.LOGIN_METHOD, "gitlab");
        expect(getLoginMethod()).toBe(LoginMethod.GITLAB);

        localStorage.setItem(LOCAL_STORAGE_KEYS.LOGIN_METHOD, "bitbucket");
        expect(getLoginMethod()).toBe(LoginMethod.BITBUCKET);

        localStorage.setItem(LOCAL_STORAGE_KEYS.LOGIN_METHOD, "azure_devops");
        expect(getLoginMethod()).toBe(LoginMethod.AZURE_DEVOPS);
      });
    });

    describe("clearLoginData", () => {
      it("removes the login method from local storage", () => {
        setLoginMethod(LoginMethod.GITHUB);
        expect(getLoginMethod()).toBe(LoginMethod.GITHUB);

        clearLoginData();
        expect(getLoginMethod()).toBeNull();
      });

      it("does not throw when no login method is set", () => {
        expect(() => clearLoginData()).not.toThrow();
      });
    });
  });

  describe("CTA utilities", () => {
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

    describe("persistence", () => {
      it("dismissed state persists across multiple reads", () => {
        setCTADismissed("homepage");

        expect(isCTADismissed("homepage")).toBe(true);
        expect(isCTADismissed("homepage")).toBe(true);
        expect(isCTADismissed("homepage")).toBe(true);
      });
    });
  });
});
