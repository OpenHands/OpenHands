import { Trans, useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

export function BitbucketDCTokenHelpAnchor() {
  const { t } = useTranslation();

  return (
    <p
      data-testid="bitbucket-dc-token-help-anchor"
      className="text-xs max-w-[680px]"
    >
      <Trans
        i18nKey={I18nKey.BITBUCKET_DATA_CENTER$TOKEN_HELP_TEXT}
        components={[
          <a
            key="bitbucket-dc-token-help-anchor-link"
            aria-label={t(I18nKey.GIT$BITBUCKET_DC_TOKEN_HELP_LINK)}
            href="https://confluence.atlassian.com/bitbucketserver/http-access-tokens-939515499.html"
            target="_blank"
            className="underline underline-offset-2"
            rel="noopener noreferrer"
          />,
        ]}
      />
    </p>
  );
}
