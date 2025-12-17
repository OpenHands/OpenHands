import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";

export const downloadConversation = async (conversationId: string) => {
  const blob = await V1ConversationService.downloadConversation(conversationId);

  // Create a download link
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `conversation_${conversationId}.zip`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};
