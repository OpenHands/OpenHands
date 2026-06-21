import React from "react";
import { useTranslation } from "react-i18next";
import { Check, Link as LinkIcon } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

interface CopyInviteLinkButtonProps {
  inviteUrl: string;
}

export function CopyInviteLinkButton({ inviteUrl }: CopyInviteLinkButtonProps) {
  const { t } = useTranslation();
  const [copied, setCopied] = React.useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(inviteUrl);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      type="button"
      data-testid="copy-invite-link-button"
      onClick={handleCopy}
      className={cn(
        "flex items-center gap-1 text-xs cursor-pointer hover:underline shrink-0",
        copied ? "text-success" : "text-primary",
      )}
    >
      {copied ? <Check size={12} /> : <LinkIcon size={12} />}
      {copied
        ? t(I18nKey.ORG$INVITE_LINK_COPIED)
        : t(I18nKey.ORG$COPY_INVITE_LINK)}
    </button>
  );
}
