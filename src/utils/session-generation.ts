/**
 * A non-reversible fingerprint of a session credential, used only to detect a
 * rotation. The credential itself must never enter a query key, cached value,
 * or long-lived map; a generation is enough to tell two sessions apart.
 *
 * FNV-1a (32-bit): collisions across one conversation's successive keys are
 * astronomically unlikely, and the value carries no recoverable key material.
 */
export function sessionGeneration(
  sessionApiKey: string | null | undefined,
): string | null {
  if (!sessionApiKey) return null;

  let hash = 2166136261;
  for (let index = 0; index < sessionApiKey.length; index += 1) {
    hash ^= sessionApiKey.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}
