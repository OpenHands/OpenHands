import { useEffect, useState } from "react";
import { ArrowLeft, ArrowRight, ExternalLink, RotateCw } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { normalizeLivePreviewUrl } from "#/utils/browser-live-url";
import { cn } from "#/utils/utils";

type BrowserChromeBarProps = {
  url: string;
  hasPage: boolean;
  canGoBack: boolean;
  canGoForward: boolean;
  canReload: boolean;
  onNavigate: (url: string) => void;
  onBack: () => void;
  onForward: () => void;
  onReload: () => void;
};

export function BrowserChromeBar({
  url,
  hasPage,
  canGoBack,
  canGoForward,
  canReload,
  onNavigate,
  onBack,
  onForward,
  onReload,
}: BrowserChromeBarProps) {
  const { t } = useTranslation("openhands");
  const [draft, setDraft] = useState(url);

  useEffect(() => {
    setDraft(url);
  }, [url]);

  const submitDraft = () => {
    const normalized = normalizeLivePreviewUrl(draft);
    if (!normalized) {
      return;
    }
    setDraft(normalized);
    onNavigate(normalized);
  };

  const navButtonClassName = (enabled: boolean) =>
    cn(
      "shrink-0 inline-flex items-center justify-center w-6 h-6 rounded-md",
      enabled
        ? "text-[var(--oh-text-tertiary)] hover:bg-tertiary cursor-pointer"
        : "text-[var(--oh-text-tertiary)] opacity-40 cursor-not-allowed",
    );

  const iconClassName = "w-3.5 h-3.5";
  const navIconStroke = 1.5;

  return (
    <div
      className="flex w-full min-h-[34px] shrink-0 items-center gap-1 border-b border-[var(--oh-border)] px-2 py-1.5"
      data-testid="browser-chrome-bar"
    >
      <div className="flex shrink-0 items-center gap-0.5">
        <button
          type="button"
          disabled={!canGoBack}
          onClick={onBack}
          aria-label={t(I18nKey.BUTTON$BACK)}
          title={t(I18nKey.BUTTON$BACK)}
          data-testid="browser-chrome-back"
          className={navButtonClassName(canGoBack)}
        >
          <ArrowLeft
            className={iconClassName}
            aria-hidden
            strokeWidth={navIconStroke}
          />
        </button>
        <button
          type="button"
          disabled={!canGoForward}
          onClick={onForward}
          aria-label={t(I18nKey.BUTTON$FORWARD)}
          title={t(I18nKey.BUTTON$FORWARD)}
          data-testid="browser-chrome-forward"
          className={navButtonClassName(canGoForward)}
        >
          <ArrowRight
            className={iconClassName}
            aria-hidden
            strokeWidth={navIconStroke}
          />
        </button>
        <button
          type="button"
          disabled={!canReload}
          onClick={onReload}
          aria-label={t(I18nKey.BUTTON$RELOAD)}
          title={t(I18nKey.BUTTON$RELOAD)}
          data-testid="browser-chrome-reload"
          className={navButtonClassName(canReload)}
        >
          <RotateCw
            className={iconClassName}
            aria-hidden
            strokeWidth={navIconStroke}
          />
        </button>
      </div>

      <form
        className="flex min-w-0 flex-1"
        onSubmit={(event) => {
          event.preventDefault();
          submitDraft();
        }}
      >
        <input
          type="text"
          inputMode="url"
          autoComplete="off"
          spellCheck={false}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={() => {
            // Keep the draft in sync with the committed URL when the user
            // abandons an incomplete edit.
            if (draft.trim() === "" || draft === url) {
              setDraft(url);
            }
          }}
          placeholder={t(I18nKey.BROWSER$URL_PLACEHOLDER)}
          aria-label={t(I18nKey.BROWSER$URL_INPUT_LABEL)}
          title={draft || undefined}
          data-testid="browser-chrome-url"
          className={cn(
            "min-h-7 min-w-0 w-full rounded-md border border-[var(--oh-border)]",
            "bg-[var(--oh-surface-raised)] px-2 text-xs leading-5 outline-none",
            "text-[var(--oh-text-tertiary)] placeholder:text-[var(--oh-text-dim)]",
            "focus:border-[var(--oh-border-strong,var(--oh-border))] focus:ring-1 focus:ring-[var(--oh-border)]",
          )}
        />
      </form>

      {hasPage && url ? (
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={t(I18nKey.BUTTON$OPEN_IN_NEW_TAB)}
          title={t(I18nKey.BUTTON$OPEN_IN_NEW_TAB)}
          data-testid="browser-chrome-open-external"
          className={cn(
            "shrink-0 inline-flex items-center justify-center w-6 h-6 rounded-md",
            "text-[var(--oh-text-tertiary)] hover:bg-tertiary cursor-pointer",
          )}
        >
          <ExternalLink className={iconClassName} aria-hidden strokeWidth={2} />
        </a>
      ) : (
        <button
          type="button"
          disabled
          aria-label={t(I18nKey.BUTTON$OPEN_IN_NEW_TAB)}
          title={t(I18nKey.BUTTON$OPEN_IN_NEW_TAB)}
          className={navButtonClassName(false)}
        >
          <ExternalLink className={iconClassName} aria-hidden strokeWidth={2} />
        </button>
      )}
    </div>
  );
}
