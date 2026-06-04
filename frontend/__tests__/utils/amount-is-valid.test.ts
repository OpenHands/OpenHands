import { describe, expect, test } from "vitest";
import {
  amountIsValid,
  getAmountValidationError,
} from "#/utils/amount-is-valid";

describe("amountIsValid", () => {
  describe("passes", () => {
    test("when an integer amount is within the accepted range", () => {
      expect(amountIsValid("10")).toBe(true);
      expect(amountIsValid("25000")).toBe(true);
    });

    test("when a valid amount has surrounding whitespace", () => {
      expect(amountIsValid(" 10 ")).toBe(true);
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
      expect(amountIsValid("1e2")).toBe(false);
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

  describe("getAmountValidationError", () => {
    test("returns the matching validation reason", () => {
      expect(getAmountValidationError("")).toBe("empty");
      expect(getAmountValidationError("abc")).toBe("invalid");
      expect(getAmountValidationError("1e2")).toBe("invalid");
      expect(getAmountValidationError("-5")).toBe("negative");
      expect(getAmountValidationError("10.50")).toBe("not_integer");
      expect(getAmountValidationError("9")).toBe("below_minimum");
      expect(getAmountValidationError("25001")).toBe("above_maximum");
      expect(getAmountValidationError("10")).toBeNull();
    });
  });
});
