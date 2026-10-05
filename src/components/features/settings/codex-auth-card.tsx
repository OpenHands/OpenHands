import React from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import {
  CodexAuthService,
  type CodexDeviceChallenge,
} from "#/api/codex-auth-service";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useCodexAuthStatus } from "#/hooks/query/use-codex-auth-status";
import {
  CODEX_AUTH_QUERY_KEYS,
  SECRETS_QUERY_KEYS,
} from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "./brand-button";
import { CopyToClipboardButton } from "#/components/shared/buttons/copy-to-clipboard-button";

export function CodexAuthCard({ enabled = true }: { enabled?: boolean }) {
  const { backend } = useActiveBackend();
  return (
    <CodexAuthCardForBackend
      key={`${backend.id}:${backend.connectionRevision}`}
      enabled={enabled}
    />
  );
}

// @spec CAO-001 — Server-owned Codex ChatGPT sign-in.
function CodexAuthCardForBackend({ enabled }: { enabled: boolean }) {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const status = useCodexAuthStatus(enabled);
  const queryClient = useQueryClient();
  const [challenge, setChallenge] = React.useState<CodexDeviceChallenge | null>(
    null,
  );
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState(false);
  const [copied, setCopied] = React.useState(false);
  const attempt = React.useRef(0);
  const handle = React.useRef<string | null>(null);
  const connected = status.data?.connected === true;
  const queryKey = React.useMemo(
    () => CODEX_AUTH_QUERY_KEYS.status(backend.id, backend.connectionRevision),
    [backend.id, backend.connectionRevision],
  );

  React.useEffect(
    () => () => {
      attempt.current += 1;
      if (handle.current)
        void CodexAuthService.cancel(backend, handle.current).catch(() => {});
    },
    [backend],
  );

  React.useEffect(() => {
    if (!challenge) return undefined;
    if (!enabled) {
      attempt.current += 1;
      handle.current = null;
      setChallenge(null);
      void CodexAuthService.cancel(backend, challenge.device_code).catch(
        () => {},
      );
      return undefined;
    }
    let stopped = false;
    let timer: number;
    const generation = attempt.current;
    const schedule = () => {
      timer = window.setTimeout(
        async () => {
          if (stopped) return;
          try {
            if (Date.now() >= challenge.expires_at) throw new Error("expired");
            const next = await CodexAuthService.poll(
              backend,
              challenge.device_code,
            );
            if (stopped || generation !== attempt.current) return;
            if (next.state === "pending") {
              schedule();
              return;
            }
            await queryClient.cancelQueries({ queryKey });
            if (stopped || generation !== attempt.current) return;
            queryClient.setQueryData(queryKey, next);
            handle.current = null;
            setChallenge(null);
            setError(!next.connected);
            void queryClient.invalidateQueries({
              queryKey: SECRETS_QUERY_KEYS.all,
            });
          } catch {
            if (stopped || generation !== attempt.current) return;
            void CodexAuthService.cancel(backend, challenge.device_code).catch(
              () => {},
            );
            handle.current = null;
            setChallenge(null);
            setError(true);
          }
        },
        Math.max(1, challenge.interval_seconds) * 1000,
      );
    };
    schedule();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [challenge, enabled, backend, queryClient, queryKey]);

  const start = async () => {
    const generation = ++attempt.current;
    setBusy(true);
    setError(false);
    try {
      const next = await CodexAuthService.start(backend);
      if (generation !== attempt.current) {
        await CodexAuthService.cancel(backend, next.device_code);
        return;
      }
      handle.current = next.device_code;
      setChallenge(next);
      setCopied(false);
      window.open(next.verification_uri, "_blank", "noopener,noreferrer");
    } catch {
      if (generation === attempt.current) setError(true);
    } finally {
      if (generation === attempt.current) setBusy(false);
    }
  };

  const cancel = async () => {
    attempt.current += 1;
    const previous = handle.current;
    handle.current = null;
    setChallenge(null);
    setBusy(true);
    try {
      if (previous) await CodexAuthService.cancel(backend, previous);
    } catch {
      setError(true);
    } finally {
      setBusy(false);
    }
  };

  const logout = async () => {
    attempt.current += 1;
    setBusy(true);
    setError(false);
    try {
      const next = await CodexAuthService.logout(backend);
      handle.current = null;
      setChallenge(null);
      await queryClient.cancelQueries({ queryKey });
      queryClient.setQueryData(queryKey, next);
      void queryClient.invalidateQueries({ queryKey: SECRETS_QUERY_KEYS.all });
    } catch {
      setError(true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section
      data-testid="codex-auth-card"
      className="flex flex-col gap-3 rounded-xl border border-border p-4"
    >
      <p role="status">
        {t(
          connected
            ? I18nKey.SETTINGS$CODEX_CONNECTED
            : I18nKey.SETTINGS$SUBSCRIPTION_STATUS_DISCONNECTED,
        )}
      </p>
      {error && <p role="alert">{t(I18nKey.SETTINGS$CODEX_AUTH_ERROR)}</p>}
      {(status.isError || backend.kind === "cloud") && (
        <p role="alert">
          {t(I18nKey.SETTINGS$SUBSCRIPTION_STATUS_UNAVAILABLE)}
        </p>
      )}
      {challenge ? (
        <>
          <p>{t(I18nKey.SETTINGS$SUBSCRIPTION_DEVICE_INSTRUCTIONS)}</p>
          <div className="flex items-center gap-2">
            <code>{challenge.user_code}</code>
            <CopyToClipboardButton
              isHidden={false}
              isDisabled={false}
              mode={copied ? "copied" : "copy"}
              onClick={() => {
                void navigator.clipboard
                  .writeText(challenge.user_code)
                  .then(() => setCopied(true))
                  .catch(() => setError(true));
              }}
            />
          </div>
          <a
            href={challenge.verification_uri}
            target="_blank"
            rel="noopener noreferrer"
          >
            {t(I18nKey.SETTINGS$SUBSCRIPTION_OPEN_LOGIN)}
          </a>
          <BrandButton
            type="button"
            variant="secondary"
            isDisabled={busy}
            onClick={cancel}
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
        </>
      ) : connected ? (
        <BrandButton
          type="button"
          variant="secondary"
          isDisabled={busy}
          onClick={logout}
        >
          {t(I18nKey.BUTTON$DISCONNECT)}
        </BrandButton>
      ) : (
        <BrandButton
          type="button"
          variant="primary"
          isDisabled={
            !enabled ||
            busy ||
            status.isLoading ||
            status.isError ||
            backend.kind === "cloud"
          }
          onClick={start}
        >
          {t(I18nKey.SETTINGS$CODEX_SIGN_IN)}
        </BrandButton>
      )}
    </section>
  );
}
