import { Blocks, Package, Workflow } from "lucide-react";
import { McpLogoBadge } from "#/components/features/mcp-logo-badge";
import { SkillIconBadge } from "#/components/features/skills/skill-icon-badge";
import { MANIFEST_ICON_BY_SLUG } from "#/components/features/manifest/manifest-icons";
import type { CatalogEntry } from "#/types/marketplaces";

interface CatalogEntryBadgeProps {
  entry: CatalogEntry;
}

/**
 * The icon slot of a catalog card, drawn the way each kind's own page draws
 * it: a skill gets the Skills badge, an MCP server its integration logo, an
 * automation template its manifest icon, and the rest their kind's icon in
 * the same chrome.
 */
export function CatalogEntryBadge({ entry }: CatalogEntryBadgeProps) {
  if (entry.kind === "skill") {
    return <SkillIconBadge skillName={entry.name} />;
  }
  if (entry.kind === "mcp") {
    return (
      <McpLogoBadge
        testId={`catalog-entry-logo-${entry.name}`}
        entry={{
          id: entry.name,
          name: entry.name,
          logoUrl: entry.icon?.logo_url ?? undefined,
          iconBg: entry.icon?.bg ?? undefined,
          iconColor: entry.icon?.color ?? undefined,
        }}
      />
    );
  }
  const slug = entry.icon?.slug as
    keyof typeof MANIFEST_ICON_BY_SLUG | undefined;
  const Icon =
    entry.kind === "automation"
      ? ((slug && MANIFEST_ICON_BY_SLUG[slug]) ?? Workflow)
      : entry.kind === "app"
        ? Blocks
        : Package;
  return (
    <McpLogoBadge
      testId={`catalog-entry-logo-${entry.name}`}
      fallback={<Icon className="h-5 w-5" strokeWidth={2} />}
    />
  );
}
