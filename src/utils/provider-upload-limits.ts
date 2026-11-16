/**
 * Maximum attachment size per LLM provider.
 *
 * The ceiling is a property of the provider API that receives the image, so it
 * is expressed per provider rather than as one global number. Provider ids are
 * the LiteLLM-style ids Canvas already uses (`src/utils/map-provider.ts`), so
 * no id mapping is needed.
 *
 * These limits describe **images**. A non-image attachment (a PDF, a video, an
 * archive) is uploaded to the workspace and read from there by the agent, so
 * the provider never receives its bytes and the provider's image limit says
 * nothing about it. Those attachments are bounded by
 * {@link DEFAULT_ATTACHMENT_LIMIT_MB} instead.
 */

/** Megabytes. */
const MB = 1024 * 1024;

/**
 * Window global the static server injects with a deployment's provider limit
 * overrides, as JSON: `{"anthropic": 10}`.
 */
const PROVIDER_LIMITS_WINDOW_KEY = "__AGENT_CANVAS_PROVIDER_UPLOAD_LIMITS__";

/**
 * Deployment-level overrides for the provider image limits, in megabytes.
 *
 * A provider listed here uses the operator's value instead of the documented
 * one, so a deployment whose provider accepts more (or less) than the public
 * limit does not need a rebuilt bundle.
 */
function getProviderLimitOverrides(): Record<string, number> {
  if (typeof window === "undefined") return {};

  const raw = (window as unknown as Record<string, unknown>)[
    PROVIDER_LIMITS_WINDOW_KEY
  ];
  if (typeof raw !== "string" || raw.length === 0) return {};

  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed))
      return {};

    const overrides: Record<string, number> = {};
    for (const [provider, value] of Object.entries(
      parsed as Record<string, unknown>,
    )) {
      if (typeof value === "number" && Number.isFinite(value) && value > 0) {
        overrides[provider] = value;
      }
    }
    return overrides;
  } catch {
    return {};
  }
}

/**
 * Fallback ceiling for a provider with no documented image limit, and the
 * ceiling for attachments that are not images.
 *
 * It matches the historical Canvas limit, so an unknown provider and a
 * non-image attachment keep today's behaviour instead of silently accepting
 * more than they used to.
 */
export const DEFAULT_ATTACHMENT_LIMIT_MB = 3;

/**
 * Largest image that can travel as an embedded (base64) attachment.
 *
 * Embedded images are base64-encoded into the chat event, which the agent
 * server forwards to the browser as a WebSocket frame. The server drops the
 * connection for a frame larger than `MAX_FRAME_BYTES` (4 MiB), and base64
 * inflates the payload by ~4/3, so ~3 MiB of raw image bytes is the ceiling
 * regardless of what the provider accepts.
 *
 * This bound applies to the embed path only. An image uploaded to the
 * workspace travels over HTTP and is bounded by the provider's limit.
 */
export const MAX_EMBEDDED_IMAGE_BYTES = 3 * MB;

/**
 * Documented per-provider image ceilings, in megabytes.
 *
 * The values are the providers' limits for an image sent inline in a request.
 */
const PROVIDER_IMAGE_LIMIT_MB: Record<string, number> = {
  anthropic: 5,
  gemini: 20,
  openai: 50,
  azure: 50,
  vertex_ai: 20,
};

/**
 * The image limit for a provider, in bytes.
 *
 * A provider with no documented limit falls back to
 * {@link DEFAULT_ATTACHMENT_LIMIT_MB}, so an incomplete table can only make
 * the limit stricter than the provider allows, never more permissive.
 */
export function getProviderImageLimitBytes(provider?: string | null): number {
  if (provider) {
    const override = getProviderLimitOverrides()[provider];
    if (override !== undefined) return override * MB;
  }

  const limitMb =
    provider && PROVIDER_IMAGE_LIMIT_MB[provider] !== undefined
      ? PROVIDER_IMAGE_LIMIT_MB[provider]
      : DEFAULT_ATTACHMENT_LIMIT_MB;

  return limitMb * MB;
}

/**
 * The image limit for a provider, in megabytes.
 */
export function getProviderImageLimitMb(provider?: string | null): number {
  return getProviderImageLimitBytes(provider) / MB;
}

/**
 * Whether a provider has a documented image limit.
 *
 * Used by the settings UI to distinguish a known ceiling from the fallback.
 */
export function hasProviderImageLimit(provider?: string | null): boolean {
  if (!provider) return false;
  if (getProviderLimitOverrides()[provider] !== undefined) return true;
  return PROVIDER_IMAGE_LIMIT_MB[provider] !== undefined;
}

/**
 * Every provider with a documented image limit, with its limit in megabytes.
 */
export function listProviderImageLimits(): {
  provider: string;
  limitMb: number;
}[] {
  return Object.entries(PROVIDER_IMAGE_LIMIT_MB).map(([provider, limitMb]) => ({
    provider,
    limitMb,
  }));
}

/**
 * Whether a file is an image, which decides which limit applies to it.
 *
 * The composer accepts every file type, so this is the boundary between the
 * provider's image limit and the generic attachment limit.
 */
export function isImageAttachment(file: File): boolean {
  return file.type.startsWith("image/");
}
