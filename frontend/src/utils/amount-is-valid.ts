export const MINIMUM_AMOUNT = 10;
export const MAXIMUM_AMOUNT = 25_000;

export type AmountValidationError =
  | "empty"
  | "invalid"
  | "negative"
  | "not_integer"
  | "below_minimum"
  | "above_maximum";

const DECIMAL_AMOUNT_PATTERN = /^-?\d+(?:\.\d+)?$/;

export const getAmountValidationError = (
  amount: string,
): AmountValidationError | null => {
  const trimmedAmount = amount.trim();
  if (!trimmedAmount) return "empty";

  if (!DECIMAL_AMOUNT_PATTERN.test(trimmedAmount)) return "invalid";

  const value = Number(trimmedAmount);
  if (!Number.isFinite(value)) return "invalid";
  if (value < 0) return "negative";
  if (trimmedAmount.includes(".")) return "not_integer";
  if (!Number.isInteger(value)) return "not_integer";
  if (value < MINIMUM_AMOUNT) return "below_minimum";
  if (value > MAXIMUM_AMOUNT) return "above_maximum";

  return null;
};

export const amountIsValid = (amount: string) =>
  getAmountValidationError(amount) === null;
