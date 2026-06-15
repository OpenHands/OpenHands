import { AxiosError } from "axios";
import { DEFAULT_CONCURRENT_SANDBOX_LIMIT } from "./constants";

interface ConcurrencyLimitErrorDetail {
  error: "CONCURRENCY_LIMIT_REACHED";
  message: string;
  limit: number;
  current: number;
}

// FastAPI wraps HTTPException detail in a "detail" field
interface FastAPIErrorResponse {
  detail: ConcurrencyLimitErrorDetail;
}

export function isConcurrencyLimitError(
  error: unknown,
): error is AxiosError<FastAPIErrorResponse> {
  if (!(error instanceof AxiosError)) return false;
  if (error.response?.status !== 429) return false;
  return error.response?.data?.detail?.error === "CONCURRENCY_LIMIT_REACHED";
}

export function getConcurrencyLimit(
  error: AxiosError<FastAPIErrorResponse>,
): number {
  return (
    error.response?.data?.detail?.limit ?? DEFAULT_CONCURRENT_SANDBOX_LIMIT
  );
}
