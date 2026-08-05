function normalizeVersion(version: string): string {
  return version.trim().replace(/^v/i, "").split(/[+-]/, 1)[0] ?? "";
}

function numericVersionParts(version: string): number[] {
  return normalizeVersion(version)
    .split(".")
    .map((part) => Number.parseInt(part, 10))
    .map((part) => (Number.isFinite(part) ? part : 0));
}

export function compareSemanticVersions(a: string, b: string): number {
  const aParts = numericVersionParts(a);
  const bParts = numericVersionParts(b);
  const maxLength = Math.max(aParts.length, bParts.length);

  for (let index = 0; index < maxLength; index += 1) {
    const aPart = aParts[index] ?? 0;
    const bPart = bParts[index] ?? 0;

    if (aPart > bPart) return 1;
    if (aPart < bPart) return -1;
  }

  return 0;
}
