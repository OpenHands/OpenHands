import { useId } from "react";
import { useTranslation } from "react-i18next";
import { Building2 } from "lucide-react";
import ExternalLinkIcon from "#/icons/external-link.svg?react";
import {
  ENTERPRISE_CALLOUTS,
  type EnterpriseCalloutId,
  type EnterpriseCalloutSurface,
} from "#/constants/enterprise-docs";
import { useEnterpriseCalloutsVisible } from "#/hooks/use-enterprise-callouts-visible";
import { useTracking } from "#/hooks/use-tracking";
import { cn } from "#/utils/utils";

/** `inline` sits in page flow; `footer` closes a dialog with a divider. */
type EnterpriseCalloutVariant = "inline" | "footer";

interface EnterpriseDocsCalloutProps {
  ctaId: EnterpriseCalloutId;
  surface: EnterpriseCalloutSurface;
  variant?: EnterpriseCalloutVariant;
  className?: string;
}

const VARIANT_CLASSES: Record<EnterpriseCalloutVariant, string> = {
  inline: "text-xs text-muted",
  footer: "border-t border-border pt-4 text-xs text-muted",
};

/**
 * A short, non-blocking note that points at the official docs for an
 * OpenHands Enterprise capability. Renders nothing on Enterprise backends.
 */
export function EnterpriseDocsCallout({
  ctaId,
  surface,
  variant = "inline",
  className,
}: EnterpriseDocsCalloutProps) {
  const { t } = useTranslation("openhands");
  const isVisible = useEnterpriseCalloutsVisible();
  const { trackEnterpriseDocsLinkClicked } = useTracking();
  const bodyId = useId();

  if (!isVisible) return null;
  const definition = ENTERPRISE_CALLOUTS[ctaId];

  return (
    <div
      data-testid={`enterprise-callout-${ctaId}`}
      data-surface={surface}
      className={cn(
        "flex items-start gap-2",
        VARIANT_CLASSES[variant],
        className,
      )}
    >
      <Building2 className="mt-0.5 size-4 shrink-0" aria-hidden />
      <p className="leading-5">
        <span id={bodyId}>{t(definition.bodyKey)}</span>{" "}
        <a
          href={definition.href}
          target="_blank"
          rel="noopener noreferrer"
          aria-describedby={bodyId}
          data-testid={`enterprise-callout-${ctaId}-link`}
          className="whitespace-nowrap text-link hover:underline"
          onClick={() => trackEnterpriseDocsLinkClicked({ ctaId, surface })}
        >
          {t(definition.linkKey)}
          <ExternalLinkIcon className="ml-1 inline size-3.5" aria-hidden />
        </a>
      </p>
    </div>
  );
}
