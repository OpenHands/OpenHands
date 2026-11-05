import { NoBackendAvailableError } from "#/api/agent-server-client-options";
import { OPENAI_SUBSCRIPTION_STATUS_PATH } from "#/constants/llm-subscription";
import { isSdkHttpError } from "#/utils/sdk-http-error";

export type SubscriptionStatusErrorKind =
  | "no-backend"
  | "missing-endpoints"
  | "unauthorized"
  | "unreachable"
  | "failed";

export interface SubscriptionStatusErrorInfo {
  kind: SubscriptionStatusErrorKind;
  detail: string;
}

export function describeSubscriptionStatusError(
  error: unknown,
): SubscriptionStatusErrorInfo {
  const request = `GET ${OPENAI_SUBSCRIPTION_STATUS_PATH}`;

  if (error instanceof NoBackendAvailableError) {
    return { kind: "no-backend", detail: "" };
  }

  if (isSdkHttpError(error)) {
    const { status } = error;
    const kind: SubscriptionStatusErrorKind =
      status === 404 || status === 405
        ? "missing-endpoints"
        : status === 401 || status === 403
          ? "unauthorized"
          : "failed";
    return { kind, detail: `${request} → HTTP ${status}` };
  }

  const cause = error instanceof Error ? error.cause : undefined;
  const transportError = cause ?? error;
  const unreachable =
    (transportError instanceof TypeError &&
      /^(failed to fetch|fetch failed|load failed|network request failed|networkerror when attempting to fetch resource\.?)$/i.test(
        transportError.message,
      )) ||
    (transportError instanceof Error && transportError.name === "TimeoutError");
  return { kind: unreachable ? "unreachable" : "failed", detail: request };
}
