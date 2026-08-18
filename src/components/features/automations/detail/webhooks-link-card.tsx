import { Link } from "react-router";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import LinkExternalIcon from "#/icons/link-external.svg?react";
import { ConfigField } from "./config-field";

/**
 * Webhook sources are org-scoped, not per-automation (the backend has no
 * automation_id on a CustomWebhook), so management lives on its own settings
 * screen rather than nested under one automation's detail page. This is a
 * discoverability link from the Advanced section, gated on the deployment
 * actually supporting custom webhook delivery.
 */
export function WebhooksLinkCard() {
  const { t } = useTranslation("openhands");

  return (
    <ConfigField
      icon={<LinkExternalIcon className="size-3.5" />}
      label={t(I18nKey.AUTOMATIONS$DETAIL$WEBHOOKS)}
    >
      <Link
        to="/automations/webhooks"
        data-testid="manage-webhooks-link"
        className="text-sm text-[var(--oh-primary)] hover:underline"
      >
        {t(I18nKey.AUTOMATIONS$DETAIL$MANAGE_WEBHOOK_SOURCES)}
      </Link>
    </ConfigField>
  );
}
