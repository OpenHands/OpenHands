import { useQuery } from "@tanstack/react-query";
import { AppLoginService } from "#/api/app-login-service";
import { APP_LOGIN_QUERY_KEYS } from "#/hooks/query/query-keys";

export function useAppLoginStatus() {
  return useQuery({
    queryKey: APP_LOGIN_QUERY_KEYS.status,
    queryFn: () => AppLoginService.getStatus(),
    staleTime: 1000 * 60,
    retry: false,
    meta: { disableToast: true },
  });
}

export function useAppLoginSession(enabled: boolean) {
  return useQuery({
    queryKey: APP_LOGIN_QUERY_KEYS.session,
    queryFn: () => AppLoginService.getSession(),
    enabled,
    staleTime: 1000 * 30,
    retry: false,
    meta: { disableToast: true },
  });
}
