// Email validation regex pattern
const EMAIL_REGEX = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

/**
 * Validates if a string is a valid email address format
 * @param email The email string to validate
 * @returns true if the email format is valid, false otherwise
 */
export const isValidEmail = (email: string): boolean => EMAIL_REGEX.test(email);

/**
 * Validates an array of email addresses and returns the invalid ones
 * @param emails Array of email strings to validate
 * @returns Array of invalid email addresses
 */
export const getInvalidEmails = (emails: string[]): string[] =>
  emails.filter((email) => !isValidEmail(email));

/**
 * Checks if all emails in an array are valid
 * @param emails Array of email strings to validate
 * @returns true if all emails are valid, false otherwise
 */
export const areAllEmailsValid = (emails: string[]): boolean =>
  emails.every((email) => isValidEmail(email));

/**
 * Checks if an array contains duplicate values (case-insensitive for emails)
 * @param values Array of strings to check
 * @returns true if duplicates exist, false otherwise
 */
export const hasDuplicates = (values: string[]): boolean => {
  const lowercased = values.map((v) => v.toLowerCase());
  return new Set(lowercased).size !== lowercased.length;
};
