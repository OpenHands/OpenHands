import { describe, expect, it } from "vitest";
import { EMAIL_REGEX, isValidEmail, validateEmail } from "#/utils/auth";

describe("auth utilities", () => {
  describe("EMAIL_REGEX", () => {
    it("should match valid email formats", () => {
      const validEmails = [
        "user@example.com",
        "user.name@example.com",
        "user+tag@example.com",
        "user123@example.org",
        "user@sub.domain.com",
        "user_name@example.io",
        "a@b.co",
        "test.email+alias@gmail.com",
        "first.last@company.co.uk",
        "user%tag@example.net",
      ];

      validEmails.forEach((email) => {
        expect(EMAIL_REGEX.test(email)).toBe(true);
      });
    });

    it("should not match invalid email formats", () => {
      const invalidEmails = [
        "",
        "not-an-email",
        "@example.com",
        "user@",
        "user@.com",
        "user@example",
        "user@example.",
        "user @example.com",
        "user@ example.com",
        "user@@example.com",
      ];

      invalidEmails.forEach((email) => {
        expect(EMAIL_REGEX.test(email)).toBe(false);
      });
    });
  });

  describe("isValidEmail", () => {
    it("should return true for valid emails", () => {
      expect(isValidEmail("user@example.com")).toBe(true);
      expect(isValidEmail("test.email@domain.org")).toBe(true);
      expect(isValidEmail("user+tag@mail.co.uk")).toBe(true);
    });

    it("should return false for invalid emails", () => {
      expect(isValidEmail("")).toBe(false);
      expect(isValidEmail("not-email")).toBe(false);
      expect(isValidEmail("@domain.com")).toBe(false);
      expect(isValidEmail("user@")).toBe(false);
    });
  });

  describe("validateEmail", () => {
    it("should return valid: true for valid email", () => {
      const result = validateEmail("user@example.com");
      expect(result.valid).toBe(true);
      expect(result.error).toBeUndefined();
    });

    it("should return error when email is empty", () => {
      const result = validateEmail("");
      expect(result.valid).toBe(false);
      expect(result.error).toBe("Email is required");
    });

    it("should return error when email format is invalid", () => {
      const result = validateEmail("invalid-email");
      expect(result.valid).toBe(false);
      expect(result.error).toBe("Invalid email format");
    });

    it("should return error for email without domain", () => {
      const result = validateEmail("user@");
      expect(result.valid).toBe(false);
      expect(result.error).toBe("Invalid email format");
    });

    it("should return error for email without user part", () => {
      const result = validateEmail("@example.com");
      expect(result.valid).toBe(false);
      expect(result.error).toBe("Invalid email format");
    });
  });
});
