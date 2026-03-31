import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { Text, Paragraph } from "#/ui/typography";
import { useGitConversationRouting } from "#/hooks/organizations/use-git-conversation-routing";
import { GitOrgRow } from "./git-org-row";

export function GitConversationRouting() {
  const { t } = useTranslation();
  const { orgs, claimOrg, disconnectOrg } = useGitConversationRouting();

  return (
    <div
      data-testid="git-conversation-routing"
      className="flex flex-col gap-3 w-full"
    >
      <Text className="text-[#fafafa] text-sm font-semibold leading-5">
        {t(I18nKey.ORG$GIT_CONVERSATION_ROUTING)}
      </Text>

      <Paragraph className="text-[#8c8c8c] text-sm font-normal leading-5">
        {t(I18nKey.ORG$GIT_CONVERSATION_ROUTING_DESCRIPTION)}
      </Paragraph>

      <div className="border border-[#242424] rounded-[6px] overflow-hidden">
        {orgs.map((org, index) => (
          <GitOrgRow
            key={org.id}
            org={org}
            isLast={index === orgs.length - 1}
            onClaim={claimOrg}
            onDisconnect={disconnectOrg}
          />
        ))}
      </div>
    </div>
  );
}
