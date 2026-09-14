import React from "react";
import { useTranslation } from "react-i18next";
import { AxiosError } from "axios";
import { I18nKey } from "#/i18n/declaration";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { BrandButton } from "#/components/features/settings/brand-button";
import McpService from "#/api/mcp-service/mcp-service.api";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

/**
 * Hostnames the agent-server accepts when relaying a callback URL — it pins
 * the redirect to its own loopback listener before fetching it.
 * `new URL("http://[::1]/").hostname` keeps the brackets, hence the third entry.
 */
const LOOPBACK_HOSTNAMES: readonly string[] = [
  "localhost",
  "127.0.0.1",
  "[::1]",
];

/**
 * The agent-server reports validation failures (400 `Invalid OAuth callback
 * URL`, 400 `Unexpected OAuth callback URL`, 409 `OAuth callback is not
 * ready`) as a FastAPI-style `{ detail: string }` body. `retrieveAxiosErrorMessage`
 * only recognises `error`/`message` fields, so it would surface the generic
 * Axios message instead of the backend's detail — read `detail` first and
 * fall back to it for anything else (e.g. network failures).
 */
function extractCallbackErrorMessage(error: unknown): string {
  if (
    error instanceof AxiosError &&
    typeof error.response?.data === "object" &&
    error.response.data !== null &&
    "detail" in error.response.data &&
    typeof (error.response.data as { detail?: unknown }).detail === "string"
  ) {
    return (error.response.data as { detail: string }).detail;
  }
  return retrieveAxiosErrorMessage(error);
}

/**
 * Fast client-side sanity check, deliberately limited to what the browser can
 * actually know: the relay target is always a loopback HTTP URL. The
 * agent-server's `_validate_callback_url` is authoritative for everything else
 * — it pins the path and port to the ones *this job* generated, values the
 * browser cannot see. Checking more here would reject URLs the backend
 * accepts, most importantly a provider denial
 * (`…/callback?error=access_denied&state=…`), which is worth relaying so the
 * job fails immediately instead of burning the full wait.
 */
function isRelayableCallbackUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return (
      url.protocol === "http:" && LOOPBACK_HOSTNAMES.includes(url.hostname)
    );
  } catch {
    return false;
  }
}

interface OauthCallbackFallbackProps {
  /** OAuth probe job the pasted URL belongs to. */
  jobId: string;
}

/**
 * Escape hatch for OAuth flows where the browser cannot reach the
 * agent-server's callback listener — the container case in issue #15430.
 * The user pastes the URL their browser failed to load, and the agent-server
 * fetches it from inside its own network namespace, completing the job.
 */
export function OauthCallbackFallback({ jobId }: OauthCallbackFallbackProps) {
  const { t } = useTranslation("openhands");
  const [isExpanded, setIsExpanded] = React.useState(false);
  const [url, setUrl] = React.useState("");
  const [error, setError] = React.useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [hasSubmitted, setHasSubmitted] = React.useState(false);

  const label = t(I18nKey.MCP$OAUTH_CALLBACK_FALLBACK_LABEL);

  const handleSubmit = () => {
    const trimmed = url.trim();
    if (!isRelayableCallbackUrl(trimmed)) {
      setError(t(I18nKey.MCP$OAUTH_CALLBACK_INVALID_URL));
      return;
    }
    setError(null);
    setIsSubmitting(true);
    void McpService.submitOAuthCallback(jobId, trimmed)
      // Success is not rendered here: the parent's still-running status poll
      // observes the completed job and closes the modal.
      .then(() => setHasSubmitted(true))
      .catch((err: unknown) => {
        setError(extractCallbackErrorMessage(err) || t(I18nKey.ERROR$GENERIC));
      })
      .finally(() => setIsSubmitting(false));
  };

  if (!isExpanded) {
    return (
      <button
        type="button"
        data-testid="mcp-oauth-callback-fallback-toggle"
        className="text-xs text-left text-[var(--oh-muted)] hover:text-white hover:underline transition-colors"
        onClick={() => setIsExpanded(true)}
      >
        {label}
      </button>
    );
  }

  return (
    <div
      data-testid="mcp-oauth-callback-fallback"
      className="flex flex-col gap-2 mt-2"
    >
      <SettingsInput
        testId="mcp-oauth-callback-url"
        name="oauth-callback-url"
        label={label}
        type="url"
        value={url}
        placeholder={t(I18nKey.MCP$OAUTH_CALLBACK_URL_PLACEHOLDER)}
        onChange={setUrl}
        isDisabled={isSubmitting || hasSubmitted}
        error={error ?? undefined}
      />
      <BrandButton
        type="button"
        variant="secondary"
        testId="mcp-oauth-callback-submit"
        className="self-end"
        isDisabled={isSubmitting || hasSubmitted || url.trim().length === 0}
        aria-busy={isSubmitting}
        onClick={handleSubmit}
      >
        {t(I18nKey.BUTTON$CONTINUE)}
      </BrandButton>
    </div>
  );
}
