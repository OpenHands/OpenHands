import { useTranslation } from "react-i18next";
import { SettingsFieldSchema } from "#/types/settings";
import { HelpLink } from "#/ui/help-link";
import { Typography } from "#/ui/typography";
import { resolveSchemaFieldDescription } from "#/utils/sdk-settings-field-metadata";

// ---------------------------------------------------------------------------
// Help links – UI-only mapping from field keys to user-facing guidance.
// Keys use conventional i18n pattern: SCHEMA$<PATH>$HELP_TEXT / HELP_LINK_TEXT
// ---------------------------------------------------------------------------
export const FIELD_HELP_LINKS: Record<
  string,
  {
    textKey: string;
    linkTextKey: string;
    href: string;
    /** Skip rendering the schema description separately when the help text already includes it. */
    hideDescription?: boolean;
    /** Optional trailing copy rendered after the link (e.g. " tab of OpenHands Cloud."). */
    suffixKey?: string;
  }
> = {
  "llm.api_key": {
    textKey: "SCHEMA$LLM$API_KEY$HELP_TEXT",
    linkTextKey: "SCHEMA$LLM$API_KEY$HELP_LINK_TEXT",
    href: "https://docs.openhands.dev/usage/local-setup#getting-an-api-key",
  },
  // Mirror the hint shown under the LLM provider's API key field when
  // OpenHands is selected as the active provider; the SDK reuses that active
  // LLM key when the critic key is empty.
  "verification.critic_api_key": {
    textKey: "SCHEMA$VERIFICATION$CRITIC_API_KEY$HELP_TEXT",
    linkTextKey: "SETTINGS$NAV_API_KEYS",
    suffixKey: "SCHEMA$VERIFICATION$CRITIC_API_KEY$HELP_SUFFIX",
    href: "https://app.all-hands.dev/settings/api-keys",
    hideDescription: true,
  },
};

export function FieldHelp({ field }: { field: SettingsFieldSchema }) {
  const { t } = useTranslation("openhands");
  const helpLink = FIELD_HELP_LINKS[field.key];
  const description = resolveSchemaFieldDescription(
    t,
    field.key,
    field.description,
  );

  return (
    <>
      {description && !helpLink?.hideDescription ? (
        <Typography.Paragraph className="text-tertiary-alt text-xs leading-5">
          {description}
        </Typography.Paragraph>
      ) : null}
      {helpLink ? (
        <HelpLink
          testId={`help-link-${field.key}`}
          text={t(helpLink.textKey)}
          linkText={t(helpLink.linkTextKey)}
          href={helpLink.href}
          suffix={helpLink.suffixKey ? ` ${t(helpLink.suffixKey)}` : undefined}
          size="settings"
          linkColor="white"
        />
      ) : null}
    </>
  );
}
