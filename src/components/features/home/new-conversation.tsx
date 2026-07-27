import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useNavigation } from "#/context/navigation-context";
import { BrandButton } from "../settings/brand-button";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { useIsCreatingConversation } from "#/hooks/use-is-creating-conversation";
import PlusIcon from "#/icons/u-plus.svg?react";

export function NewConversation() {
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

  return (
    <section className="w-full min-h-[286px] md:min-h-auto flex flex-col rounded-[12px] p-[20px] gap-[10px] border border-[var(--oh-border)] bg-[var(--oh-surface)] relative">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-[10px]">
          <PlusIcon width={17} height={14} />
          <span className="leading-5 font-bold text-base text-white">
            {t(I18nKey.COMMON$START_FROM_SCRATCH)}
          </span>
        </div>
      </div>
      <div>
        <span className="leading-[22px] text-sm font-normal text-white">
          {t(I18nKey.HOME$NEW_PROJECT_DESCRIPTION)}
        </span>
      </div>
      <BrandButton
        testId="launch-new-conversation-button"
        variant="primary"
        type="button"
        onClick={() =>
          createConversation(
            { entryPoint: "home_start_from_scratch" },
            {
              onSuccess: (data) =>
                navigate(`/conversations/${data.conversation_id}`),
            },
          )
        }
        isDisabled={isCreatingConversation}
        className="w-auto absolute bottom-5 left-5 right-5"
      >
        {!isCreatingConversation && t(I18nKey.COMMON$NEW_CONVERSATION)}
        {isCreatingConversation && t(I18nKey.HOME$LOADING)}
      </BrandButton>
    </section>
  );
}
