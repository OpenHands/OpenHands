import type { ComponentType, ReactNode, SVGProps } from "react";
import { Bot } from "lucide-react";
import type { IntegrationCatalogEntry } from "@openhands/extensions/integrations";
import { cn } from "#/utils/utils";

type McpLogoEntry = Pick<
  IntegrationCatalogEntry,
  "id" | "name" | "iconBg" | "iconColor" | "logoUrl"
>;

export type { McpLogoEntry };

interface McpLogoBadgeProps {
  entry?: McpLogoEntry | null;
  size?: "xs" | "sm" | "base" | "md";
  className?: string;
  fallback?: ReactNode;
  testId?: string;
}

const sizeClassNames = {
  xs: "h-4 w-4 rounded [&>svg]:h-2.5 [&>svg]:w-2.5",
  sm: "h-5 w-5 rounded-md [&>svg]:h-3 [&>svg]:w-3",
  base: "h-7 w-7 rounded-lg [&>svg]:h-3.5 [&>svg]:w-3.5",
  md: "h-10 w-10 rounded-lg [&>svg]:h-5 [&>svg]:w-5",
};

// Catalog entries whose remote logoUrl is unreliable render a bundled mark
// instead, keyed by IntegrationCatalogEntry.id. Currently empty — entries
// are added here only when a catalog logo 404s.
const LOCAL_LOGO_ICONS: Record<
  string,
  ComponentType<SVGProps<SVGSVGElement>>
> = {};

export function McpLogoBadge({
  entry,
  size = "md",
  className,
  fallback,
  testId,
}: McpLogoBadgeProps) {
  const LocalLogoIcon = entry ? LOCAL_LOGO_ICONS[entry.id] : undefined;
  return (
    <span
      aria-hidden="true"
      title={entry?.name}
      data-testid={testId}
      className={cn(
        "mcp-logo-badge inline-flex shrink-0 items-center justify-center overflow-hidden",
        "border border-white/10 shadow-[inset_0_1px_0_rgba(255,255,255,0.18)]",
        sizeClassNames[size],
        className,
      )}
      style={{
        backgroundColor: entry?.iconBg ?? "var(--oh-color-tertiary)",
        color: entry?.iconColor ?? "var(--oh-foreground)",
      }}
    >
      {LocalLogoIcon ? (
        <LocalLogoIcon />
      ) : entry?.logoUrl ? (
        <img
          src={entry.logoUrl}
          alt={`${entry.name} logo`}
          className="h-full w-full object-contain p-[22%]"
          onError={(e) => {
            const image = e.currentTarget;
            image.style.display = "none";
          }}
        />
      ) : (
        (fallback ?? <Bot className="h-5 w-5" strokeWidth={2.25} />)
      )}
    </span>
  );
}
