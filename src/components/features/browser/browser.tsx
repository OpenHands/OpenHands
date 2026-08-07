import { useCallback, useRef } from "react";
import { useTranslation } from "react-i18next";
import { BrowserSnapshot } from "./browser-snapshot";
import { BrowserChromeBar } from "./browser-chrome-bar";
import { EmptyBrowserMessage } from "./empty-browser-message";
import { I18nKey } from "#/i18n/declaration";
import { useBrowserStore } from "#/stores/browser-store";
import {
  tryIframeGoBack,
  tryIframeGoForward,
  tryIframeReload,
  tryReadIframeHref,
} from "#/utils/browser-iframe-navigation";

const LIVE_IFRAME_SANDBOX =
  "allow-scripts allow-same-origin allow-forms allow-popups allow-popups-to-escape-sandbox";

export function BrowserPanel() {
  const { t } = useTranslation("openhands");
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const {
    mode,
    url,
    iframeSrc,
    screenshotSrc,
    history,
    historyIndex,
    reloadToken,
    setLiveUrl,
    syncLiveUrl,
    goBack,
    goForward,
    reload,
  } = useBrowserStore();
  const hasPage = mode === "live" || mode === "screenshot";
  // Prefer the iframe's own session history (covers in-iframe link clicks).
  // Buttons stay enabled whenever a live page is up so history.back/forward
  // can no-op safely when there is nowhere to go.
  const canGoBack = mode === "live" && Boolean(iframeSrc);
  const canGoForward = mode === "live" && Boolean(iframeSrc);
  const canReload = mode === "live" && Boolean(iframeSrc);

  const imgSrc = screenshotSrc?.startsWith("data:image/png;base64,")
    ? screenshotSrc
    : `data:image/png;base64,${screenshotSrc ?? ""}`;

  const handleBack = useCallback(() => {
    // Parent-driven address-bar history takes priority: assigning a new
    // iframe `src` replaces the iframe session, so in-iframe history alone
    // cannot undo typed navigations.
    if (historyIndex > 0) {
      goBack();
      return;
    }
    tryIframeGoBack(iframeRef.current);
  }, [goBack, historyIndex]);

  const handleForward = useCallback(() => {
    if (historyIndex >= 0 && historyIndex < history.length - 1) {
      goForward();
      return;
    }
    tryIframeGoForward(iframeRef.current);
  }, [goForward, history, historyIndex]);

  const handleReload = useCallback(() => {
    if (tryIframeReload(iframeRef.current)) {
      return;
    }
    reload();
  }, [reload]);

  const handleIframeLoad = useCallback(() => {
    const href = tryReadIframeHref(iframeRef.current);
    if (href) {
      syncLiveUrl(href);
    }
  }, [syncLiveUrl]);

  return (
    <div className="flex h-full min-h-0 w-full flex-col text-[var(--oh-muted)]">
      <BrowserChromeBar
        url={url}
        hasPage={hasPage}
        canGoBack={canGoBack}
        canGoForward={canGoForward}
        canReload={canReload}
        onNavigate={setLiveUrl}
        onBack={handleBack}
        onForward={handleForward}
        onReload={handleReload}
      />
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden bg-[var(--oh-surface)]">
        {mode === "live" && iframeSrc ? (
          <iframe
            ref={iframeRef}
            // Only remount on explicit reload fallback — keep the same
            // browsing context so history.back/forward keep working after
            // in-iframe link clicks.
            key={reloadToken}
            title={t(I18nKey.BROWSER$LIVE_PREVIEW_TITLE)}
            src={iframeSrc}
            sandbox={LIVE_IFRAME_SANDBOX}
            onLoad={handleIframeLoad}
            data-testid="browser-live-iframe"
            className="h-full min-h-0 w-full flex-1 border-0 bg-white"
          />
        ) : mode === "screenshot" && screenshotSrc ? (
          <BrowserSnapshot src={imgSrc} />
        ) : (
          <EmptyBrowserMessage />
        )}
      </div>
    </div>
  );
}
