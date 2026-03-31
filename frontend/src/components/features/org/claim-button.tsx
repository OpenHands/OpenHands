import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import type { GitOrg } from "#/hooks/organizations/use-git-conversation-routing";

type ButtonState =
  | "claiming"
  | "disconnecting"
  | "disconnect"
  | "claimed"
  | "unclaimed";

const BUTTON_STYLES: Record<ButtonState, string> = {
  claiming:
    "bg-[#050505] border border-[#242424] text-[#fafafa] opacity-50 cursor-not-allowed flex items-center justify-center",
  disconnecting:
    "bg-[#050505] border border-[#242424] text-[#fafafa] opacity-50 cursor-not-allowed",
  disconnect:
    "bg-[rgba(244,63,94,0.15)] border border-[rgba(244,63,94,0.6)] text-[#fda4af] font-medium cursor-pointer",
  claimed:
    "bg-[rgba(16,185,129,0.2)] border border-[rgba(16,185,129,0.6)] text-[#6ee7b7] font-medium cursor-pointer flex items-center justify-center",
  unclaimed:
    "bg-[#050505] border border-[#242424] text-[#fafafa] cursor-pointer flex items-center justify-center",
};

const BUTTON_HOVER_STYLES: Partial<Record<ButtonState, string>> = {
  unclaimed: "bg-[rgba(31,31,31,0.6)]",
};

const BUTTON_LABELS: Record<ButtonState, I18nKey> = {
  claiming: I18nKey.ORG$CLAIM,
  disconnecting: I18nKey.ORG$DISCONNECT,
  disconnect: I18nKey.ORG$DISCONNECT,
  claimed: I18nKey.ORG$CLAIMED,
  unclaimed: I18nKey.ORG$CLAIM,
};

export function getButtonState(
  status: GitOrg["status"],
  isHovered: boolean,
): ButtonState {
  if (status === "claiming" || status === "disconnecting") return status;
  if (status === "claimed" && isHovered) return "disconnect";
  return status;
}

interface ClaimButtonProps {
  org: GitOrg;
  onClaim: (id: string) => void;
  onDisconnect: (id: string) => void;
}

export function ClaimButton({ org, onClaim, onDisconnect }: ClaimButtonProps) {
  const { t } = useTranslation();
  const [isHovered, setIsHovered] = React.useState(false);

  const buttonState = getButtonState(org.status, isHovered);
  const isDisabled =
    org.status === "claiming" || org.status === "disconnecting";

  const handleClick = () => {
    if (org.status === "unclaimed") onClaim(org.id);
    if (org.status === "claimed") onDisconnect(org.id);
  };

  return (
    <button
      type="button"
      data-testid={`claim-button-${org.id}`}
      onClick={handleClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      disabled={isDisabled}
      className={cn(
        "h-[28px] rounded px-[13px] text-xs leading-4 text-center whitespace-nowrap transition-colors",
        BUTTON_STYLES[buttonState],
        isHovered && BUTTON_HOVER_STYLES[buttonState],
      )}
    >
      {t(BUTTON_LABELS[buttonState])}
    </button>
  );
}
