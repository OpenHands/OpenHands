import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Monitor } from "lucide-react";
import {
  DesktopRequestError,
  DesktopService,
} from "#/api/integrations/desktop-service";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { LoadingSpinner } from "#/components/shared/loading-spinner";

type DesktopViewState =
  | { kind: "loading" }
  | { kind: "idle"; unavailable: boolean }
  | { kind: "starting" }
  | { kind: "ready"; url: string }
  | { kind: "error"; message: string; unavailable?: boolean };

const IFRAME_SANDBOX =
  "allow-scripts allow-same-origin allow-forms allow-popups allow-downloads";

export function DesktopPanel() {
  const { t } = useTranslation("openhands");
  const [view, setView] = useState<DesktopViewState>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        // GET /status also mints the desktop auth cookie when the session key
        // is valid, so a ready iframe can load without a prior POST /start.
        const status = await DesktopService.getStatus();
        if (cancelled) return;
        if (status.ready) {
          setView({
            kind: "ready",
            url: status.url || DesktopService.iframePath(),
          });
          return;
        }
        setView({ kind: "idle", unavailable: status.unavailable });
      } catch {
        if (!cancelled) {
          // Treat unexpected probe failures as unavailable so we never show a
          // misleading "Open Desktop" CTA against a backend without the proxy.
          setView({ kind: "idle", unavailable: true });
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const startDesktop = async () => {
    setView({ kind: "starting" });
    try {
      const status = await DesktopService.start();
      if (!status.ready) {
        setView({
          kind: "error",
          message: status.unavailable
            ? t(I18nKey.DESKTOP$UNAVAILABLE)
            : t(I18nKey.DESKTOP$FAILED),
          unavailable: status.unavailable,
        });
        return;
      }
      setView({
        kind: "ready",
        url: status.url || DesktopService.iframePath(),
      });
    } catch (err) {
      const unavailable =
        err instanceof DesktopRequestError ? err.unavailable : false;
      setView({
        kind: "error",
        message: unavailable
          ? t(I18nKey.DESKTOP$UNAVAILABLE)
          : t(I18nKey.DESKTOP$FAILED),
        unavailable,
      });
    }
  };

  if (view.kind === "ready") {
    return (
      <div className="flex h-full min-h-0 flex-col" data-testid="desktop-panel">
        <iframe
          title={t(I18nKey.COMMON$DESKTOP)}
          data-testid="desktop-iframe"
          src={view.url}
          className="h-full w-full flex-1 border-0 bg-black"
          sandbox={IFRAME_SANDBOX}
          allow="clipboard-read; clipboard-write"
        />
      </div>
    );
  }

  return (
    <div
      className="flex h-full min-h-0 flex-col items-center justify-center gap-4 p-6 text-center"
      data-testid="desktop-panel"
    >
      <Monitor className="h-10 w-10 text-[var(--oh-muted)]" aria-hidden />
      {view.kind === "loading" && (
        <LoadingSpinner size="small" data-testid="desktop-status-spinner" />
      )}
      {view.kind === "starting" && (
        <>
          <LoadingSpinner size="small" data-testid="desktop-start-spinner" />
          <p className="text-sm text-[var(--oh-muted)]">
            {t(I18nKey.DESKTOP$STARTING)}
          </p>
        </>
      )}
      {(view.kind === "idle" || view.kind === "error") && (
        <>
          <p
            className="max-w-sm text-sm text-[var(--oh-muted)]"
            data-testid="desktop-status-message"
          >
            {view.kind === "error"
              ? view.message
              : view.unavailable
                ? t(I18nKey.DESKTOP$UNAVAILABLE)
                : t(I18nKey.DESKTOP$OPEN)}
          </p>
          {!(view.kind === "idle" && view.unavailable) &&
            !(view.kind === "error" && view.unavailable) && (
              <button
                type="button"
                data-testid="desktop-open-button"
                onClick={() => void startDesktop()}
                className={cn(
                  "flex h-9 items-center justify-center rounded bg-white px-4 text-sm font-medium text-black",
                  "cursor-pointer transition-opacity hover:opacity-90",
                )}
              >
                {t(I18nKey.DESKTOP$OPEN)}
              </button>
            )}
        </>
      )}
    </div>
  );
}
