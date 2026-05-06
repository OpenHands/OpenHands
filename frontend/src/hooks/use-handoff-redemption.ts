import { useEffect, useState } from "react";

import AuthService from "#/api/auth-service/auth-service.api";

const HANDOFF_PARAM = "handoff_code";

// Module-level dedupe so React 19 StrictMode's double-effect doesn't redeem
// the same single-use code twice (the second call returns 410 and causes the
// redeem to look failed). Both mounts share the same in-flight Promise.
const inFlight = new Map<string, Promise<void>>();

/**
 * Detects ?handoff_code=<code> on initial page load, redeems it for a session
 * cookie at our origin, then performs a full navigation to the param-stripped
 * URL so the SPA reloads with the cookie already attached to every request.
 *
 * Why full navigation instead of in-place refetch: Chromium has a timing
 * window where a Partitioned cookie set on an XHR response isn't yet attached
 * to immediately-following XHRs in the same page lifetime. The post-redeem
 * /api/authenticate call ends up cookie-less and the SPA bounces to /login.
 * A full navigation matches what the production middleware does (302 to the
 * clean URL) and avoids the race entirely.
 *
 * In production the FastAPI middleware redeems before the SPA loads, so the
 * param is gone and this hook is a no-op. In dev (Vite serves the HTML) the
 * middleware never sees the request, so this hook is the redemption path.
 */
export const useHandoffRedemption = () => {
  const [isRedeeming, setIsRedeeming] = useState(() => {
    if (typeof window === "undefined") return false;
    return new URLSearchParams(window.location.search).has(HANDOFF_PARAM);
  });

  useEffect(() => {
    if (!isRedeeming) return undefined;

    const code = new URLSearchParams(window.location.search).get(HANDOFF_PARAM);
    if (!code) {
      setIsRedeeming(false);
      return undefined;
    }

    const cleanUrl = (() => {
      const p = new URLSearchParams(window.location.search);
      p.delete(HANDOFF_PARAM);
      const search = p.toString();
      return (
        window.location.pathname +
        (search ? `?${search}` : "") +
        window.location.hash
      );
    })();

    let cancelled = false;

    // De-dupe across StrictMode double-mounts: both invocations await the
    // same single redeem network call.
    let promise = inFlight.get(code);
    if (!promise) {
      promise = (async () => {
        try {
          await AuthService.redeemHandoffCode(code);
        } catch {
          // Swallow — bad/expired/already-used codes fall through to sign-in.
        }
      })();
      inFlight.set(code, promise);
    }

    (async () => {
      await promise;
      if (cancelled) return;
      // Full navigation to the clean URL so the next page load picks up the
      // cookie. replace() so the dirty URL doesn't sit in history.
      window.location.replace(cleanUrl);
    })();

    return () => {
      cancelled = true;
    };
  }, [isRedeeming]);

  return { isRedeeming };
};
