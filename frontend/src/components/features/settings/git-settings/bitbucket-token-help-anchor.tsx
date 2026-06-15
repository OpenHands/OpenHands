import { Trans, useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

export function BitbucketTokenHelpAnchor() {
  const { t } = useTranslation();

  return (
    <p data-testid="bitbucket-token-help-anchor" className="text-xs">
      <Trans
        i18nKey={I18nKey.BITBUCKET$TOKEN_HELP_TEXT}
        components={[
          <a
            key="bitbucket-token-help-anchor-link"
            aria-label={t(I18nKey.GIT$BITBUCKET_TOKEN_HELP_LINK)}
            href="https://bitbucket.org/account/settings/api-tokens/"
            target="_blank"
            className="underline underline-offset-2"
            rel="noopener noreferrer"
          />,
          <a
            key="bitbucket-token-help-anchor-link-2"
            aria-label={t(I18nKey.GIT$BITBUCKET_TOKEN_SEE_MORE_LINK)}
            href="https://support.atlassian.com/bitbucket-cloud/docs/api-tokens/"
            target="_blank"
            className="underline underline-offset-2"
            rel="noopener noreferrer"
          />,
        ]}
      />
    </p>
  );
}
