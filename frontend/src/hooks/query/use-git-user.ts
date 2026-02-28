import { useQuery } from "@tanstack/react-query";
import React from "react";
import { usePostHog } from "posthog-js/react";
import { useConfig } from "./use-config";
import UserService from "#/api/user-service/user-service.api";
import { useShouldShowUserFeatures } from "#/hooks/use-should-show-user-features";
import { useLogout } from "../mutation/use-logout";

export const useGitUser = () => {
  const posthog = usePostHog();
  const { data: config } = useConfig();
  const logout = useLogout();

  // Use the shared hook to determine if we should fetch user data
  const shouldFetchUser = useShouldShowUserFeatures();

  const user = useQuery({
    queryKey: ["user"],
    queryFn: UserService.getUser,
    enabled: shouldFetchUser,
    retry: false,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });

  React.useEffect(() => {
    if (user.data) {
      posthog.identify(user.data.login, {
        company: user.data.company,
        name: user.data.name,
        email: user.data.email,
        user: user.data.login,
        mode: config?.app_mode || "oss",
      });
    }
  }, [user.data]);

  // If we get a 401 here, it means that the integration tokens need to be
  // refreshed. Since this happens at login, we log out.
  React.useEffect(() => {
    if (user?.error?.response?.status === 401) {
      logout.mutate();
    }
  }, [user.status]);

  return user;
};
