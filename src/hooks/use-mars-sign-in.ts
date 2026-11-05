import React from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";

import {
  getMarsErrorMessage,
  requireMarsBridge,
} from "#/api/mars/mars-tunnel-backend";
import { MARS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";

/**
 * DigitalOcean sign-in by OAuth or personal access token, plus switching
 * between saved connections and signing out. The token goes straight to the
 * main process, which verifies it against harness-api before keeping it, so a
 * success here means the team really has Managed Agents.
 */
export function useMarsSignIn({
  onAuthChanged,
}: { onAuthChanged?: () => void } = {}) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const [error, setError] = React.useState<string | null>(null);

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

  const oauth = useMutation({
    mutationFn: () => requireMarsBridge().signInWithOAuth(),
    onMutate,
    onSuccess,
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

  return {
    signInWithOAuth: () => oauth.mutate(),
    saveToken: (token: string) => pat.mutate(token),
    switchConnection: (id: string) => switchConnection.mutate(id),
    signOut: () => signOut.mutate(),
    isSigningIn: oauth.isPending,
    isSavingToken: pat.isPending,
    isSigningOut: signOut.isPending,
    error,
    clearError: () => setError(null),
  };
}
