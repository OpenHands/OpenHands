import { describe, expect, test } from "vitest";
import { amountIsValid } from "#/utils/amount-is-valid";

describe("amountIsValid", () => {
  describe("passes", () => {
    test("when an integer amount is within the accepted range", () => {
      expect(amountIsValid("10")).toBe(true);
      expect(amountIsValid("25000")).toBe(true);
    });
  });

  describe("fails", () => {
    test("when the amount is negative", () => {
      expect(amountIsValid("-5")).toBe(false);
      expect(amountIsValid("-25")).toBe(false);
    });

    test("when the amount is zero", () => {
      expect(amountIsValid("0")).toBe(false);
    });

    test("when an empty string is passed", () => {
      expect(amountIsValid("")).toBe(false);
      expect(amountIsValid("     ")).toBe(false);
    });

    test("when a non-numeric value is passed", () => {
      expect(amountIsValid("abc")).toBe(false);
      expect(amountIsValid("1abc")).toBe(false);
      expect(amountIsValid("abc1")).toBe(false);
    });

    test("when a non-integer numeric value is passed", () => {
      expect(amountIsValid("10.0")).toBe(false);
      expect(amountIsValid("10.50")).toBe(false);
    });

    test("when an amount less than the minimum is passed", () => {
      // test assumes the minimum is 10
      expect(amountIsValid("9")).toBe(false);
      expect(amountIsValid("9.99")).toBe(false);
    });

    test("when an amount more than the maximum is passed", () => {
      // test assumes the minimum is 25000
      expect(amountIsValid("25001")).toBe(false);
      expect(amountIsValid("25000.01")).toBe(false);
    });
  });
});
