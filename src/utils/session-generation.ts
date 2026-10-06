/**
 * A keyed fingerprint of a session credential, used only to detect a rotation.
 * The credential itself must never enter a query key, cached value, or
 * long-lived map.
 *
 * A stable unkeyed hash (e.g. FNV-1a) would let anyone who obtains the
 * fingerprint test candidate credentials offline, which matters when the key is
 * guessable. Instead the value is an HMAC keyed with a random per-process secret
 * that is never persisted or exposed, so a leaked fingerprint reveals nothing
 * about the credential and cannot be matched against guesses.
 */
let keyMaterial: Uint8Array<ArrayBuffer> | null = null;
let hmacKey: Promise<CryptoKey> | null = null;

function getHmacKey(): Promise<CryptoKey> | null {
  if (!globalThis.crypto?.subtle) return null;
  keyMaterial ??= crypto.getRandomValues(new Uint8Array(32));
  hmacKey ??= crypto.subtle.importKey(
    "raw",
    keyMaterial,
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  return hmacKey;
}

/**
 * 128-bit hex tag; collisions across one conversation's keys are negligible.
 * Returns null when no key is present or the runtime has no Web Crypto (a
 * non-secure context), so callers degrade to not detecting rotation rather than
 * falling back to a guessable unkeyed hash.
 */
export async function sessionGeneration(
  sessionApiKey: string | null | undefined,
): Promise<string | null> {
  if (!sessionApiKey) return null;

  const key = getHmacKey();
  if (!key) return null;

  try {
    const signature = await crypto.subtle.sign(
      "HMAC",
      await key,
      new TextEncoder().encode(sessionApiKey),
    );
    const bytes = new Uint8Array(signature, 0, 16);
    return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join(
      "",
    );
  } catch {
    return null;
  }
}
