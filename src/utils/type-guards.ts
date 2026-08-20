import { AxiosError } from "axios";

export const isAxiosErrorWithErrorField = (
  error: AxiosError,
): error is AxiosError<{ error: string }> =>
  typeof error.response?.data === "object" &&
  error.response?.data !== null &&
  "error" in error.response.data &&
  typeof error.response?.data?.error === "string";

export const isAxiosErrorWithMessageField = (
  error: AxiosError,
): error is AxiosError<{ message: string }> =>
  typeof error.response?.data === "object" &&
  error.response?.data !== null &&
  "message" in error.response.data &&
  typeof error.response?.data?.message === "string";
