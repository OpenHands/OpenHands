import { useCallback, useEffect, useMemo, useRef } from "react";
import { SERVER_CONNECTION_ERROR_MESSAGE } from "#/constants/server-connection-error";
import { useErrorMessageStore } from "#/stores/error-message-store";

const CONNECTION_ERROR_GRACE_MS = 5_000;

export function useConnectionErrorGracePeriod(identity: string | null) {
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const cancel = useCallback(() => {
    if (timer.current !== null) clearTimeout(timer.current);
    timer.current = null;
  }, []);

  const report = useCallback(() => {
    // Repeated retry failures must not postpone an ongoing outage forever.
    if (timer.current !== null || document.visibilityState === "hidden") return;
    timer.current = setTimeout(() => {
      const store = useErrorMessageStore.getState();
      if (
        document.visibilityState !== "hidden" &&
        store.errorType !== "conversation"
      ) {
        store.setErrorMessage(SERVER_CONNECTION_ERROR_MESSAGE, "connection");
      }
    }, CONNECTION_ERROR_GRACE_MS);
  }, []);

  useEffect(() => {
    const onVisibilityChange = () => {
      if (document.visibilityState === "hidden") cancel();
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      cancel();
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [identity, cancel]);

  return useMemo(() => ({ report, cancel }), [report, cancel]);
}
