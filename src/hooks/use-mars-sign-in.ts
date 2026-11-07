import React from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";

import {
  getMarsErrorMessage,
  requireMarsBridge,
  type MarsAuthState,
} from "#/api/mars/mars-tunnel-backend";
import { MARS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";

const MIN_AWAIT_INTERVAL_MS = 1_000;

/** A browser sign-in waiting on DigitalOcean. */
export interface PendingMarsSignIn {
  /** DigitalOcean's page, offered again if the new tab did not open. */
  authorizeUrl: string;
  /** False when the return address will not load here, so it is pasted back. */
  returnsHere: boolean;
}

/**
 * DigitalOcean sign-in by OAuth or personal access token, plus switching
 * between saved connections and signing out. Tokens and OAuth proofs stay in
 * the bridge (the desktop main process or the Canvas server), which verifies
 * a credential against harness-api before keeping it, so a success here
 * means the team really has Managed Agents.
 */
export function useMarsSignIn({
  onAuthChanged,
}: { onAuthChanged?: () => void } = {}) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const [error, setError] = React.useState<string | null>(null);
  const [pending, setPending] = React.useState<PendingMarsSignIn | null>(null);
  /** The browser sign-in this page is waiting on, if any. */
  const attemptRef = React.useRef<string | null>(null);
  /** The tab opened for DigitalOcean's page. */
  const tabRef = React.useRef<Window | null>(null);

  // Leaving stops the waiting, not the sign-in: a return that lands later
  // still signs the host in, and the next view shows it.
  React.useEffect(
    () => () => {
      attemptRef.current = null;
    },
    [],
  );

  // A new attempt drops the previous one's error (a failed sign-out included,
  // which would otherwise linger on the sign-in form).
  const onMutate = () => setError(null);
  const onSuccess = () => {
    setError(null);
    void queryClient.invalidateQueries({ queryKey: MARS_QUERY_KEYS.all });
    onAuthChanged?.();
  };
  const onError = (failure: unknown) =>
    setError(
      getMarsErrorMessage(failure) ??
        t(I18nKey.BACKEND$DIGITALOCEAN_AUTH_FAILED),
    );

  /**
   * The page's sign-in: DigitalOcean opens in the tab the click already
   * made, and the bridge reports when the return reached the Canvas server
   * (or the user pasted it). Resolves null when the sign-in was cancelled.
   */
  const signInInBrowser = async (): Promise<MarsAuthState | null> => {
    const bridge = requireMarsBridge();
    let attempt;
    try {
      attempt = await bridge.beginSignIn!();
    } catch (failure) {
      tabRef.current?.close();
      throw failure;
    }
    const id = attempt.attemptId;
    attemptRef.current = id;
    setPending({
      authorizeUrl: attempt.authorizeUrl,
      returnsHere: attempt.returnsHere !== false,
    });
    const tab = tabRef.current;
    if (tab && !tab.closed && attempt.authorizeUrl.startsWith("https://")) {
      // DigitalOcean's page gets no handle back to Canvas.
      tab.opener = null;
      tab.location.href = attempt.authorizeUrl;
    }
    for (;;) {
      const askedAt = Date.now();
      const result = await bridge.awaitSignIn!(id);
      if (attemptRef.current !== id) return null;
      if (result.status === "done") return result.authState;
      if (result.status === "failed") {
        throw new Error(result.message ?? "DigitalOcean sign-in failed.");
      }
      // The bridge holds each call open; one that answers at once is not
      // asked again straight away.
      const wait = MIN_AWAIT_INTERVAL_MS - (Date.now() - askedAt);
      if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
    }
  };

  const oauth = useMutation({
    mutationFn: async () => {
      const bridge = requireMarsBridge();
      if (!bridge.beginSignIn || !bridge.awaitSignIn) {
        return bridge.signInWithOAuth();
      }
      try {
        return await signInInBrowser();
      } finally {
        setPending(null);
        tabRef.current = null;
      }
    },
    onMutate,
    // A cancelled sign-in changed nothing; the form it was started from stays.
    onSuccess: (result) => {
      if (result) onSuccess();
    },
    onError,
  });
  const pastedReturn = useMutation({
    mutationFn: (returned: string) =>
      requireMarsBridge().completeSignIn!(
        returned.trim(),
        attemptRef.current ?? undefined,
      ),
    onMutate,
    onError,
  });
  const pat = useMutation({
    mutationFn: (token: string) => requireMarsBridge().savePat({ token }),
    onMutate,
    onSuccess,
    onError,
  });
  const switchConnection = useMutation({
    mutationFn: (id: string) => requireMarsBridge().setActiveConnection(id),
    onMutate,
    onSuccess,
    onError,
  });
  const signOut = useMutation({
    mutationFn: () => requireMarsBridge().signOut(),
    onMutate,
    onSuccess,
    onError,
  });

  const signInWithOAuth = () => {
    // A tab opened after an await counts as a popup and is blocked, so open
    // it now, inside the click, and point it at DigitalOcean once the
    // bridge has begun.
    tabRef.current = requireMarsBridge().beginSignIn
      ? window.open("about:blank", "_blank")
      : null;
    oauth.mutate();
  };

  const cancelSignIn = () => {
    const id = attemptRef.current;
    attemptRef.current = null;
    setPending(null);
    // DigitalOcean's tab stays: once cut loose from Canvas, Canvas cannot
    // close it.
    if (id) void requireMarsBridge().cancelSignIn?.(id);
  };

  return {
    signInWithOAuth,
    /** The browser sign-in in progress, for the paste-back field. */
    pendingSignIn: pending,
    completeSignIn: (returned: string) => pastedReturn.mutate(returned),
    cancelSignIn,
    saveToken: (token: string) => {
      // A token instead: the browser sign-in still waiting is dropped.
      if (attemptRef.current) cancelSignIn();
      pat.mutate(token);
    },
    switchConnection: (id: string) => switchConnection.mutate(id),
    signOut: () => signOut.mutate(),
    isSigningIn: oauth.isPending,
    isCompletingSignIn: pastedReturn.isPending,
    isSavingToken: pat.isPending,
    isSigningOut: signOut.isPending,
    error,
    clearError: () => setError(null),
  };
}
