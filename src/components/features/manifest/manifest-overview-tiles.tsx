import type { LucideIcon } from "lucide-react";
import { extensionModuleCardSurfaceClassName } from "#/utils/extension-module-card-classes";
import { cn } from "#/utils/utils";

export interface OverviewTileView {
  key: string;
  label: string;
  value: string;
  detail: string;
  Icon: LucideIcon;
}

interface ManifestOverviewTilesProps {
  /** Names the section for assistive technology. */
  label: string;
  tiles: OverviewTileView[];
}

/**
 * A grid of summary tiles. Purely presentational — which tiles exist, their
 * captions, and their computed values arrive fully resolved from the caller.
 */
export function ManifestOverviewTiles({
  label,
  tiles,
}: ManifestOverviewTilesProps) {
  return (
    <section
      aria-label={label}
      className="grid grid-cols-2 gap-3 xl:grid-cols-4"
    >
      {tiles.map((tile) => (
        <div
          key={tile.key}
          data-testid={`overview-tile-${tile.key}`}
          className={cn(extensionModuleCardSurfaceClassName, "p-4")}
        >
          <div className="flex items-center justify-between gap-2">
            <span className="truncate text-xs text-muted">{tile.label}</span>
            <tile.Icon
              className="size-4 shrink-0 text-[var(--oh-muted)]"
              aria-hidden
            />
          </div>
          <div className="mt-2 text-xl font-semibold text-content">
            {tile.value}
          </div>
          <div className="mt-1 truncate text-xs text-muted">{tile.detail}</div>
        </div>
      ))}
    </section>
  );
}
