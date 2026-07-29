export const isNonEmptyString = (value: unknown): value is string =>
  typeof value === "string" && value.length > 0;

export const pickFirstString = (...values: unknown[]): string | undefined =>
  values.find(isNonEmptyString);

export const pickFirstBoolean = (...values: unknown[]): boolean | undefined =>
  values.find((value): value is boolean => typeof value === "boolean");

export const pickFirstNumber = (...values: unknown[]): number | undefined =>
  values.find((value): value is number => typeof value === "number");

export const pickNullableString = (
  ...values: unknown[]
): string | null | undefined => {
  for (const value of values) {
    if (typeof value === "string") {
      return value;
    }
    if (value === null) {
      return null;
    }
  }

  return undefined;
};
