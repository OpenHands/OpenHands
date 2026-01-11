import { Trans } from "react-i18next";
import { Link } from "react-router";
import i18n from "#/i18n";

interface ErrorMessageBannerProps {
  message: string;
}

export function ErrorMessageBanner({ message }: ErrorMessageBannerProps) {
  return (
    <div className="w-full rounded-lg p-2 text-content border border-danger bg-danger">
      {i18n.exists(message) ? (
        <Trans
          i18nKey={message}
          components={{
            a: (
              <Link
                className="underline font-bold cursor-pointer"
                to="/settings/billing"
              >
                link
              </Link>
            ),
          }}
        />
      ) : (
        message
      )}
    </div>
  );
}
