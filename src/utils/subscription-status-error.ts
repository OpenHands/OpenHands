import { OPENAI_SUBSCRIPTION_STATUS_PATH } from "#/constants/llm-subscription";
import { isSdkHttpError } from "#/utils/sdk-http-error";

export type SubscriptionStatusErrorKind =
  | "missing-endpoints"
  | "unauthorized"
  | "unreachable"
  | "failed";

export interface SubscriptionStatusErrorInfo {
  kind: SubscriptionStatusErrorKind;
  /** Technical one-liner naming the exact failed request, for reports. */
  detail: string;
}

interface SdkHttpErrorLike {
  status: number;
  statusText?: string;
  detail?: string | undefined;
}

const MAX_DETAIL_LENGTH = 200;

function truncate(text: string): string {
  return text.length > MAX_DETAIL_LENGTH
    ? `${text.slice(0, MAX_DETAIL_LENGTH - 1)}…`
    : text;
}

/**
 * Classify a failure of the OpenAI subscription status check so the UI can
 * name the actual problem. Any request failure used to render one
 * "upgrade the agent server" message, which is only true when the server
 * genuinely lacks the subscription endpoints (404/405) — on a current server
 * the same message masked reachability, auth, and server-side errors.
 */
export function describeSubscriptionStatusError(
  error: unknown,
): SubscriptionStatusErrorInfo {
  const request = `GET ${OPENAI_SUBSCRIPTION_STATUS_PATH}`;

  if (isSdkHttpError(error)) {
    const { status, statusText, detail } = error as SdkHttpErrorLike;
    const kind: SubscriptionStatusErrorKind =
      status === 404 || status === 405
        ? "missing-endpoints"
        : status === 401 || status === 403
          ? "unauthorized"
          : "failed";
    const outcome = `HTTP ${status}${statusText ? ` ${statusText}` : ""}`;
    return {
      kind,
      detail: truncate(
        detail
          ? `${request} → ${outcome}: ${detail}`
          : `${request} → ${outcome}`,
      ),
    };
  }

  const message = error instanceof Error ? error.message : String(error);
  return { kind: "unreachable", detail: truncate(`${request} → ${message}`) };
}
