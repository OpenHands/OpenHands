import { useMutation, useQueryClient } from "@tanstack/react-query";
import { usePostHog } from "posthog-js/react";
import AuthService from "#/api/auth-service/auth-service.api";
import { useConfig } from "../query/use-config";
import { clearLoginData } from "#/utils/local-storage";

export const useLogout = () => {
  const posthog = usePostHog();
  const queryClient = useQueryClient();
  const { data: config } = useConfig();

  return useMutation({
    mutationFn: () => AuthService.logout(config?.APP_MODE ?? "oss"),
    onSuccess: async () => {
      queryClient.removeQueries({ queryKey: ["tasks"] });
      queryClient.removeQueries({ queryKey: ["settings"] });
      queryClient.removeQueries({ queryKey: ["user"] });
      queryClient.removeQueries({ queryKey: ["secrets"] });

      // Clear login method and last page from local storage
      if (config?.APP_MODE === "saas") {
        clearLoginData();
      }

      posthog.reset();

      // For B1 mode, redirect to login page; otherwise reload
      if (config?.APP_MODE === "b1") {
        window.location.href = "/login";
      } else {
        window.location.reload();
      }
    },
  });
};
