import React from "react";
import { useTranslation } from "react-i18next";
import { MicroagentStatus } from "#/types/microagent-status";
import { SuccessIndicator } from "../success-indicator";
import { Spinner } from "#/ui/spinner";
import { Typography } from "#/ui/typography";

interface MicroagentStatusIndicatorProps {
  status: MicroagentStatus;
  conversationId?: string;
  prUrl?: string;
}

export function MicroagentStatusIndicator({
  status,
  conversationId,
  prUrl,
}: MicroagentStatusIndicatorProps) {
  const { t } = useTranslation();

  const getStatusText = () => {
    switch (status) {
      case MicroagentStatus.WAITING:
        return t("MICROAGENT$STATUS_WAITING");
      case MicroagentStatus.CREATING:
        return t("MICROAGENT$STATUS_CREATING");
      case MicroagentStatus.COMPLETED:
        // If there's a PR URL, show "View your PR" instead of the default completed message
        return prUrl
          ? t("MICROAGENT$VIEW_YOUR_PR")
          : t("MICROAGENT$STATUS_COMPLETED");
      case MicroagentStatus.ERROR:
        return t("MICROAGENT$STATUS_ERROR");
      default:
        return "";
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case MicroagentStatus.WAITING:
        return <Spinner size="sm" />;
      case MicroagentStatus.CREATING:
        return <Spinner size="sm" />;
      case MicroagentStatus.COMPLETED:
        return <SuccessIndicator status="success" />;
      case MicroagentStatus.ERROR:
        return <SuccessIndicator status="error" />;
      default:
        return null;
    }
  };

  const statusText = getStatusText();
  const shouldShowAsLink = !!conversationId;
  const shouldShowPRLink = !!prUrl;

  const renderStatusText = () => {
    if (shouldShowPRLink) {
      return (
        <a
          href={prUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="underline"
        >
          {statusText}
        </a>
      );
    }

    if (shouldShowAsLink) {
      return (
        <a
          href={`/conversations/${conversationId}`}
          target="_blank"
          rel="noopener noreferrer"
          className="underline"
        >
          {statusText}
        </a>
      );
    }

    return (
      <Typography.Text className="underline">{statusText}</Typography.Text>
    );
  };

  return (
    <div className="flex items-center gap-2 mt-2 p-2 text-sm">
      {getStatusIcon()}
      {renderStatusText()}
    </div>
  );
}
