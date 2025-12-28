import { useMutation } from "@tanstack/react-query";
import EventService from "#/api/event-service/event-service.api";
import type { ConfirmationResponseRequest } from "#/api/event-service/event-service.types";

interface UseRespondToConfirmationVariables {
  conversationId: string;
  conversationUrl: string;
  sessionApiKey?: string | null;
  accept: boolean;
}

export const useRespondToConfirmation = () =>
  useMutation({
    mutationKey: ["respond-to-confirmation"],
    mutationFn: async ({
      conversationId,
      conversationUrl,
      sessionApiKey,
      accept,
    }: UseRespondToConfirmationVariables) => {
      const request: ConfirmationResponseRequest = {
        accept,
      };

      return EventService.respondToConfirmation(
        conversationId,
        conversationUrl,
        request,
        sessionApiKey,
      );
    },
  });
