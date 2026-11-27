import { useQuery } from "@tanstack/react-query";

export const useIsAuthed = () =>
  useQuery({
    queryKey: ["user", "authenticated"],
    queryFn: async () => true,
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
    retry: false,
    meta: {
      disableToast: true,
    },
  });
