import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import PlusIcon from "#/icons/u-plus.svg?react";
import { CardTitle } from "#/ui/card-title";
import { Typography } from "#/ui/typography";
import { CreateConversationButton } from "./create-conversation-button";
import { Card } from "#/ui/card";

export function NewConversation() {
  const { t } = useTranslation();

  return (
    <Card className="flex-col p-5 gap-2.5 min-h-[286px] md:min-h-auto w-full">
      <CardTitle icon={<PlusIcon width={17} height={14} />}>
        {t(I18nKey.COMMON$START_FROM_SCRATCH)}
      </CardTitle>
      <Typography.Text>
        {t(I18nKey.HOME$NEW_PROJECT_DESCRIPTION)}
      </Typography.Text>
      <CreateConversationButton />
    </Card>
  );
}
