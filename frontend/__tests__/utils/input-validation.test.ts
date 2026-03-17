import { describe, expect, test } from "vitest";
import {
  isValidEmail,
  getInvalidEmails,
  areAllEmailsValid,
  hasDuplicates,
} from "#/utils/input-validation";

describe("isValidEmail", () => {
  describe("valid email formats", () => {
    test("accepts standard email formats", () => {
      expect(isValidEmail("user@example.com")).toBe(true);
      expect(isValidEmail("john.doe@company.org")).toBe(true);
      expect(isValidEmail("test@subdomain.domain.com")).toBe(true);
    });

    test("accepts emails with numbers", () => {
      expect(isValidEmail("user123@example.com")).toBe(true);
      expect(isValidEmail("123user@example.com")).toBe(true);
      expect(isValidEmail("user@example123.com")).toBe(true);
    });

    test("accepts emails with special characters in local part", () => {
      expect(isValidEmail("user.name@example.com")).toBe(true);
      expect(isValidEmail("user+tag@example.com")).toBe(true);
      expect(isValidEmail("user_name@example.com")).toBe(true);
      expect(isValidEmail("user-name@example.com")).toBe(true);
      expect(isValidEmail("user%tag@example.com")).toBe(true);
    });

    test("accepts emails with various TLDs", () => {
      expect(isValidEmail("user@example.io")).toBe(true);
      expect(isValidEmail("user@example.co.uk")).toBe(true);
      expect(isValidEmail("user@example.travel")).toBe(true);
    });
  });

  describe("invalid email formats", () => {
    test("rejects empty strings", () => {
      expect(isValidEmail("")).toBe(false);
    });

    test("rejects strings without @", () => {
      expect(isValidEmail("userexample.com")).toBe(false);
      expect(isValidEmail("user.example.com")).toBe(false);
    });

    test("rejects strings without domain", () => {
      expect(isValidEmail("user@")).toBe(false);
      expect(isValidEmail("user@.com")).toBe(false);
    });

    test("rejects strings without local part", () => {
      expect(isValidEmail("@example.com")).toBe(false);
    });

    test("rejects strings without TLD", () => {
      expect(isValidEmail("user@example")).toBe(false);
      expect(isValidEmail("user@example.")).toBe(false);
    });

    test("rejects strings with single character TLD", () => {
      expect(isValidEmail("user@example.c")).toBe(false);
    });

    test("rejects plain text", () => {
      expect(isValidEmail("test")).toBe(false);
      expect(isValidEmail("just some text")).toBe(false);
    });

    test("rejects emails with spaces", () => {
      expect(isValidEmail("user @example.com")).toBe(false);
      expect(isValidEmail("user@ example.com")).toBe(false);
      expect(isValidEmail(" user@example.com")).toBe(false);
      expect(isValidEmail("user@example.com ")).toBe(false);
    });

    test("rejects emails with multiple @ symbols", () => {
      expect(isValidEmail("user@@example.com")).toBe(false);
      expect(isValidEmail("user@domain@example.com")).toBe(false);
    });
  });
});

describe("getInvalidEmails", () => {
  test("returns empty array when all emails are valid", () => {
    const emails = ["user@example.com", "test@domain.org"];
    expect(getInvalidEmails(emails)).toEqual([]);
  });

  test("returns all invalid emails", () => {
    const emails = ["valid@example.com", "invalid", "test@", "another@valid.org"];
    expect(getInvalidEmails(emails)).toEqual(["invalid", "test@"]);
  });

  test("returns all emails when none are valid", () => {
    const emails = ["invalid", "also-invalid", "no-at-symbol"];
    expect(getInvalidEmails(emails)).toEqual(emails);
  });

  test("handles empty array", () => {
    expect(getInvalidEmails([])).toEqual([]);
  });

  test("handles array with single invalid email", () => {
    expect(getInvalidEmails(["invalid"])).toEqual(["invalid"]);
  });

  test("handles array with single valid email", () => {
    expect(getInvalidEmails(["valid@example.com"])).toEqual([]);
  });
});

describe("areAllEmailsValid", () => {
  test("returns true when all emails are valid", () => {
    const emails = ["user@example.com", "test@domain.org", "admin@company.io"];
    expect(areAllEmailsValid(emails)).toBe(true);
  });

  test("returns false when any email is invalid", () => {
    const emails = ["user@example.com", "invalid", "test@domain.org"];
    expect(areAllEmailsValid(emails)).toBe(false);
  });

  test("returns false when all emails are invalid", () => {
    const emails = ["invalid", "also-invalid"];
    expect(areAllEmailsValid(emails)).toBe(false);
  });

  test("returns true for empty array", () => {
    expect(areAllEmailsValid([])).toBe(true);
  });

  test("returns true for single valid email", () => {
    expect(areAllEmailsValid(["valid@example.com"])).toBe(true);
  });

  test("returns false for single invalid email", () => {
    expect(areAllEmailsValid(["invalid"])).toBe(false);
  });
});

describe("hasDuplicates", () => {
  test("returns false when all values are unique", () => {
    expect(hasDuplicates(["a@test.com", "b@test.com", "c@test.com"])).toBe(
      false,
    );
  });

  test("returns true when duplicates exist", () => {
    expect(hasDuplicates(["a@test.com", "b@test.com", "a@test.com"])).toBe(true);
  });

  test("returns true for case-insensitive duplicates", () => {
    expect(hasDuplicates(["User@Test.com", "user@test.com"])).toBe(true);
    expect(hasDuplicates(["A@EXAMPLE.COM", "a@example.com"])).toBe(true);
  });

  test("returns false for empty array", () => {
    expect(hasDuplicates([])).toBe(false);
  });

  test("returns false for single item array", () => {
    expect(hasDuplicates(["single@test.com"])).toBe(false);
  });

  test("handles multiple duplicates", () => {
    expect(
      hasDuplicates(["a@test.com", "a@test.com", "b@test.com", "b@test.com"]),
    ).toBe(true);
  });
});
