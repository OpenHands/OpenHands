import { AxiosError } from "axios";
import {
  isAxiosErrorWithErrorField,
  isAxiosErrorWithMessageField,
} from "./type-guards";
import { getUserFacingConnectionErrorMessage } from "./user-facing-error";

function isAxiosError(error: unknown): error is AxiosError {
  return (
    typeof error === "object" &&
    error !== null &&
    "isAxiosError" in error &&
    (error as { isAxiosError?: unknown }).isAxiosError === true
  );
}

/**
 * Retrieve the error message from an Axios error
 * @param error The error to render a toast for
 */
export const retrieveAxiosErrorMessage = (error: unknown): string => {
  let errorMessage: string | null = null;
  let shouldPreferExtractedMessage = false;

  if (isAxiosError(error)) {
    shouldPreferExtractedMessage = true;
    if (isAxiosErrorWithErrorField(error) && error.response?.data.error) {
      errorMessage = error.response?.data.error;
    } else if (
      isAxiosErrorWithMessageField(error) &&
      error.response?.data.message
    ) {
      errorMessage = error.response?.data.message;
    } else {
      errorMessage = error.message;
    }
  } else if (error instanceof Error) {
    errorMessage = error.message;
  } else if (typeof error === "string") {
    errorMessage = error;
  } else {
    errorMessage = null;
  }

  const userFacingMessage = getUserFacingConnectionErrorMessage(
    shouldPreferExtractedMessage ? (errorMessage ?? error) : error,
  );
  return userFacingMessage ?? errorMessage ?? "";
};
