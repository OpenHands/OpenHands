import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AppLoginService } from "#/api/app-login-service";
import { APP_LOGIN_QUERY_KEYS } from "#/hooks/query/query-keys";

export function useAppLoginUsers(enabled: boolean) {
  return useQuery({
    queryKey: APP_LOGIN_QUERY_KEYS.users,
    queryFn: () => AppLoginService.listUsers(),
    enabled,
    meta: { disableToast: true },
  });
}

export function useCreateAppLoginUser() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      username,
      password,
    }: {
      username: string;
      password: string;
    }) => AppLoginService.createUser(username, password),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: APP_LOGIN_QUERY_KEYS.users,
      });
    },
  });
}

export function useDeleteAppLoginUser() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (username: string) => AppLoginService.deleteUser(username),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: APP_LOGIN_QUERY_KEYS.users,
      });
    },
  });
}
