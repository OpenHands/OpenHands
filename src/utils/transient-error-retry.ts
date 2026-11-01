import { AxiosError } from "axios";

export function isTransientRequestError(error: unknown): boolean {
  const status =
    error instanceof AxiosError
      ? (error.response?.status ?? error.status)
      : error && typeof error === "object" && "status" in error
        ? error.status
        : undefined;

  if (typeof status === "number") {
    return [408, 429, 500, 502, 503, 504].includes(status);
  }

  // Fetch rejects with TypeError; Axios identifies network failures separately.
  return (
    error instanceof TypeError ||
    (error instanceof AxiosError &&
      ["ERR_NETWORK", "ECONNABORTED", "ETIMEDOUT"].includes(error.code ?? ""))
  );
}

export const retryTransientRequest = (failureCount: number, error: unknown) =>
  failureCount < 3 && isTransientRequestError(error);
