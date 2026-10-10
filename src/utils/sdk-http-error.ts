/**
 * Whether `error` is the shared TypeScript client's `HttpError` (the server
 * answered with a non-2xx status). Matches on shape rather than `instanceof`
 * so it also recognizes errors from another copy of the client.
 */
export function isSdkHttpError(error: unknown) {
  return (
    error instanceof Error &&
    error.name === "HttpError" &&
    "status" in error &&
    typeof error.status === "number"
  );
}
