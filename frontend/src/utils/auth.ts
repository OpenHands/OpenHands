import { I18nKey } from "#/i18n/declaration";

// Email validation regex (matches backend)
export const EMAIL_REGEX = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

export const isValidEmail = (email: string): boolean => EMAIL_REGEX.test(email);

export const validateEmail = (
  email: string,
): { valid: boolean; error?: string } => {
  if (!email) {
    return { valid: false, error: I18nKey.AUTH$EMAIL_REQUIRED };
  }
  if (!isValidEmail(email)) {
    return { valid: false, error: I18nKey.AUTH$INVALID_EMAIL_FORMAT };
  }
  return { valid: true };
};
