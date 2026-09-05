import type { SetupFieldType } from "#/manifests/types";

/**
 * Shared Review + Import row order.
 *
 * Only a few types are promoted or demoted. Everything else keeps the order
 * the form (or import spec) already declared.
 *
 * Bubble up: identity, when it runs, where it runs.
 * Leave in place: short config (text, select, and unknown names).
 * Sink: long body (textarea, prompt), then plugins.
 */
export type PreviewFieldKind =
  | SetupFieldType
  | "name"
  | "trigger"
  | "prompt"
  | "plugins";

const IDENTITY_NAMES = new Set(["name", "title"]);

export function isPreviewIdentityField(name: string): boolean {
  return IDENTITY_NAMES.has(name);
}

function previewFieldRank(name: string, kind?: PreviewFieldKind): number {
  if (isPreviewIdentityField(name) || kind === "name") return 0;

  if (kind === "cron" || kind === "trigger" || name === "schedule") return 10;
  if (kind === "timezone" || name === "timezone") return 11;

  if (kind === "repo-picker" || name.includes("repositor")) return 20;

  if (kind === "prompt" || name === "prompt") return 80;
  if (kind === "textarea") return 80;

  if (kind === "plugins" || name === "plugins") return 90;

  return 50;
}

export function sortPreviewFields<T>(
  fields: T[],
  getName: (item: T) => string,
  getKind?: (item: T) => PreviewFieldKind | undefined,
): T[] {
  return fields
    .map((item, index) => ({ item, index }))
    .sort((a, b) => {
      const rank =
        previewFieldRank(getName(a.item), getKind?.(a.item)) -
        previewFieldRank(getName(b.item), getKind?.(b.item));
      return rank || a.index - b.index;
    })
    .map(({ item }) => item);
}
