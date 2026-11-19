import { getApiErrorBody } from "#/utils/api-error-message";
import { isSdkHttpError } from "#/utils/sdk-http-error";

/**
 * The `exception` in the 500 an Agent Server (1.61 and earlier) answers with
 * when one of a conversation's event files is empty or cut short, e.g. after
 * the machine went down mid-write. Servers that include
 * software-agent-sdk#6669 serve a stand-in event instead.
 */
const UNREADABLE_EVENT_FILE =
  /^Missing event file: |validation error for Event\b[\s\S]*Invalid JSON/;

/**
 * The server's reason when `error` is a request that failed because an event
 * file is unreadable, otherwise null. Such a server cannot open the
 * conversation at all: history, count and the session socket fail alike.
 */
export function getUnreadableEventFileReason(error: unknown): string | null {
  if (!isSdkHttpError(error) || (error as { status?: number }).status !== 500) {
    return null;
  }
  const body = getApiErrorBody(error);
  if (!body || typeof body !== "object") return null;
  const { exception } = body as { exception?: unknown };
  return typeof exception === "string" && UNREADABLE_EVENT_FILE.test(exception)
    ? exception
    : null;
}
