import { useNavigation } from "#/contexts/navigation-context";

export function useOptionalConversationId() {
  const { conversationId } = useNavigation();

  return { conversationId };
}

export function useConversationId() {
  const { conversationId } = useOptionalConversationId();

  if (!conversationId) {
    throw new Error(
      "useConversationId must be used within a route that has a conversationId parameter",
    );
  }

  return { conversationId };
}
