// Email validation regex (matches backend)
export const EMAIL_REGEX = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

export const isValidEmail = (email: string): boolean => EMAIL_REGEX.test(email);

export const validateEmail = (
  email: string,
): { valid: boolean; error?: string } => {
  if (!email) {
    return { valid: false, error: "Email is required" };
  }
  if (!isValidEmail(email)) {
    return { valid: false, error: "Invalid email format" };
  }
  return { valid: true };
};
