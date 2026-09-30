import React from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";

import {
  getMarsBridge,
  getMarsErrorMessage,
} from "#/api/mars/mars-tunnel-backend";
import { MARS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";

/**
 * DigitalOcean sign-in by OAuth or personal access token. The token goes
 * straight to the main process, which verifies it against harness-api before
 * keeping it, so a success here means the team really has Managed Agents.
 */
export function useMarsSignIn({
  onSignedIn,
}: { onSignedIn?: () => void } = {}) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const [error, setError] = React.useState<string | null>(null);

  const onSuccess = () => {
    setError(null);
    void queryClient.invalidateQueries({ queryKey: MARS_QUERY_KEYS.all });
    onSignedIn?.();
  };
  const onError = (failure: unknown) =>
    setError(
      getMarsErrorMessage(failure) ??
        t(I18nKey.BACKEND$DIGITALOCEAN_AUTH_FAILED),
    );

  const oauth = useMutation({
    mutationFn: () => getMarsBridge()!.signInWithOAuth(),
    onSuccess,
    onError,
  });
  const pat = useMutation({
    mutationFn: (token: string) => getMarsBridge()!.savePat({ token }),
    onSuccess,
    onError,
  });

  return {
    signInWithOAuth: () => oauth.mutate(),
    saveToken: (token: string) => pat.mutate(token),
    isSigningIn: oauth.isPending,
    isSavingToken: pat.isPending,
    error,
    clearError: () => setError(null),
  };
}
