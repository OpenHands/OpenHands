import { useTranslation } from "react-i18next";
import { BrandButton } from "../../settings/brand-button";
import { useNavigation } from "#/context/navigation-context";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { useIsCreatingConversation } from "#/hooks/use-is-creating-conversation";
import { I18nKey } from "#/i18n/declaration";

export function CreateConversationButton() {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();

  const {
    mutate: createConversation,
    isPending,
    isSuccess,
  } = useCreateConversation();
  const isCreatingConversationElsewhere = useIsCreatingConversation();

  // We check for isSuccess because the app might require time to render
  // into the new conversation screen after the conversation is created.
  const isCreatingConversation =
    isPending || isSuccess || isCreatingConversationElsewhere;

  const handleCreateConversation = () => {
    createConversation(
      { entryPoint: "home_new_conversation_button" },
      {
        onSuccess: (data) => navigate(`/conversations/${data.conversation_id}`),
      },
    );
  };

  return (
    <BrandButton
      testId="launch-new-conversation-button"
      variant="primary"
      type="button"
      onClick={handleCreateConversation}
      isDisabled={isCreatingConversation}
      className="w-auto absolute bottom-5 left-5 right-5"
    >
      {!isCreatingConversation && t(I18nKey.COMMON$NEW_CONVERSATION)}
      {isCreatingConversation && t(I18nKey.HOME$LOADING)}
    </BrandButton>
  );
}
