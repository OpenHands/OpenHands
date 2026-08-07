/**
 * Accepts only http(s) URLs for the live Browser iframe. Rejects javascript:,
 * data:, and other schemes that would be unsafe in an iframe src.
 */
export function isValidLivePreviewUrl(
  value: string | null | undefined,
): boolean {
  if (!value) {
    return false;
  }
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

function looksLikeLocalHost(host: string): boolean {
  const hostname = host.split("/")[0]?.split(":")[0]?.toLowerCase() ?? "";
  return (
    hostname === "localhost" ||
    hostname === "127.0.0.1" ||
    hostname === "[::1]" ||
    hostname === "::1"
  );
}

/**
 * Normalize a user-typed address bar value into an http(s) URL.
 * Bare hosts get https:// (or http:// for localhost / loopback).
 */
export function normalizeLivePreviewUrl(raw: string): string | null {
  const trimmed = raw.trim();
  if (!trimmed) {
    return null;
  }

  let candidate = trimmed;
  // Require "://" so host:port values like "localhost:8089" are not
  // mistaken for a URI scheme (e.g. the bogus scheme "localhost:").
  if (!/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(trimmed)) {
    const scheme = looksLikeLocalHost(trimmed) ? "http" : "https";
    candidate = `${scheme}://${trimmed}`;
  }

  return isValidLivePreviewUrl(candidate) ? candidate : null;
}
