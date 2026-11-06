import { Blocks, Package, Workflow, type LucideIcon } from "lucide-react";
import { INTEGRATION_CATALOG } from "@openhands/extensions/integrations";
import { McpLogoBadge } from "#/components/features/mcp-logo-badge";
import { McpLogoStackBadge } from "#/components/features/mcp-page/mcp-logo-stack-badge";
import { SkillIconBadge } from "#/components/features/skills/skill-icon-badge";
import { MANIFEST_ICON_BY_SLUG } from "#/components/features/manifest/manifest-icons";
import { getMarketplaceEntryById } from "#/utils/mcp-marketplace-utils";
import type { CatalogEntry } from "#/types/marketplaces";

interface CatalogEntryBadgeProps {
  entry: CatalogEntry;
}

const KIND_ICONS: Partial<Record<CatalogEntry["kind"], LucideIcon>> = {
  app: Blocks,
  plugin: Package,
  automation: Workflow,
};

/**
 * The icon slot, drawn as each kind's own page draws it: the Skills badge for
 * a skill, the integration logo for an MCP server, the manifest icon (or the
 * logos of the integrations it needs) for an automation template, and the
 * kind's icon in the same chrome for the rest.
 */
export function CatalogEntryBadge({ entry }: CatalogEntryBadgeProps) {
  const testId = `catalog-entry-logo-${entry.kind}-${entry.name}`;

  if (entry.kind === "skill") {
    return <SkillIconBadge skillName={entry.name} />;
  }

  if (entry.kind === "mcp") {
    return (
      <McpLogoBadge
        testId={testId}
        entry={{
          id: entry.name,
          name: entry.title,
          logoUrl: entry.icon?.logo_url ?? undefined,
          iconBg: entry.icon?.bg ?? undefined,
          iconColor: entry.icon?.color ?? undefined,
        }}
      />
    );
  }

  if (entry.kind === "automation") {
    const slug = entry.icon?.slug;
    const Icon = slug
      ? (MANIFEST_ICON_BY_SLUG as Record<string, LucideIcon | undefined>)[slug]
      : undefined;
    const required = entry.requires.flatMap((id) => {
      const integration = getMarketplaceEntryById(id, INTEGRATION_CATALOG);
      return integration ? [integration] : [];
    });
    if (!Icon && required.length > 0) {
      return <McpLogoStackBadge entries={required} testId={testId} />;
    }
    const Fallback = Icon ?? Workflow;
    return (
      <McpLogoBadge
        entry={null}
        testId={testId}
        fallback={<Fallback className="h-5 w-5" strokeWidth={2.25} />}
      />
    );
  }

  const Icon = KIND_ICONS[entry.kind] ?? Package;
  return (
    <McpLogoBadge
      entry={null}
      testId={testId}
      fallback={<Icon className="h-5 w-5" strokeWidth={2.25} />}
    />
  );
}
