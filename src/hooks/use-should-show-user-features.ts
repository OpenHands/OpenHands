import React from "react";
import { useIsAuthed } from "./query/use-is-authed";
import { useUserProviders } from "./use-user-providers";

export const useShouldShowUserFeatures = (): boolean => {
  const { data: isAuthed } = useIsAuthed();
  const { providers } = useUserProviders();

  return React.useMemo(() => {
    if (!isAuthed) return false;
    return providers.length > 0;
  }, [isAuthed, providers.length]);
};
