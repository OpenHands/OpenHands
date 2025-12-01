import React from "react";
import { useTranslation } from "react-i18next";
import ArrowDown from "#/icons/angle-down-solid.svg?react";
import ArrowUp from "#/icons/angle-up-solid.svg?react";
import i18n from "#/i18n";
import { MarkdownRenderer } from "../markdown/markdown-renderer";

interface ErrorMessageProps {
  errorId?: string;
  defaultMessage: string;
}

export function ErrorMessage({ errorId, defaultMessage }: ErrorMessageProps) {
  const { t } = useTranslation();
  const [showDetails, setShowDetails] = React.useState(false);

  const hasValidTranslationId = !!errorId && i18n.exists(errorId);
  const errorKey = hasValidTranslationId
    ? errorId
    : "CHAT_INTERFACE$AGENT_ERROR_MESSAGE";

  return (
    <div className="flex flex-col gap-2 border-l-2 pl-2 my-2 py-2 border-danger text-sm w-full">
      <div className="font-bold text-danger">
        {t(errorKey)}
        <button
          type="button"
          onClick={() => setShowDetails((prev) => !prev)}
          className="cursor-pointer text-left"
        >
          {showDetails ? (
            <ArrowUp className="h-4 w-4 ml-2 inline fill-danger" />
          ) : (
            <ArrowDown className="h-4 w-4 ml-2 inline fill-danger" />
          )}
        </button>
      </div>

      {showDetails && <MarkdownRenderer>{defaultMessage}</MarkdownRenderer>}
    </div>
  );
}
