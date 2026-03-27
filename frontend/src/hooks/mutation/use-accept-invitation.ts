import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AxiosError } from "axios";
import { organizationService } from "#/api/organization-service/organization-service.api";

interface AcceptInvitationError {
  detail: string;
}

export const useAcceptInvitation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ token }: { token: string }) =>
      organizationService.acceptInvitation({ token }),
    onSuccess: () => {
      // Invalidate organizations query to refresh the list
      queryClient.invalidateQueries({ queryKey: ["organizations"] });
    },
    // Note: Error handling is done in the component to allow for custom messages
    // based on the error code
  });
};

/**
 * Extract the error code from an invitation acceptance error.
 * @param error The Axios error from the accept invitation mutation
 * @returns The error code string (e.g., 'invitation_expired', 'already_member')
 */
export const getInvitationErrorCode = (
  error: AxiosError<AcceptInvitationError>,
): string | null => {
  if (error.response?.data?.detail) {
    return error.response.data.detail;
  }
  return null;
};
