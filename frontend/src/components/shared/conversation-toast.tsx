import toast from "react-hot-toast";

export function renderConversationErroredToast(
  _conversationId: string,
  message: string,
): void {
  toast.error(message);
}

export function renderConversationCreatedToast(): void {
  toast.success("Runtime started");
}

export function renderConversationFinishedToast(): void {
  toast.success("Conversation finished");
}

export function renderConversationStartingToast(conversationId: string): void {
  toast.loading("Starting runtime...", {
    id: `starting-${conversationId}`,
  });
}
