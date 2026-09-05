import type { PreviewFieldKind } from "./automation-preview-order";

/**
 * Discrete token lists are chipped. Prose and single facts stay plain text.
 *
 * Chip: repositories, plugins, feeds, topics.
 * Do not chip: name, schedule, timezone, trigger, environment, webhook,
 * prompt, notes, summary.
 */
export function shouldChipPreviewField(
  name: string,
  kind?: PreviewFieldKind,
): boolean {
  return (
    kind === "repo-picker" ||
    kind === "plugins" ||
    name === "feeds" ||
    name === "topics"
  );
}

export function previewChipItems(
  name: string,
  kind: PreviewFieldKind | undefined,
  parts: string[],
): string[] | undefined {
  if (!shouldChipPreviewField(name, kind)) return undefined;

  const tokens =
    kind === "repo-picker" || kind === "plugins"
      ? parts
      : name === "feeds"
        ? parts.flatMap((part) => part.split("\n"))
        : parts.flatMap((part) => part.split(/[\n,]+/));

  const items = tokens.map((token) => token.trim()).filter(Boolean);
  return items.length > 0 ? items : undefined;
}
