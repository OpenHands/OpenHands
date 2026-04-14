import React from "react";
import { FaArrowRotateRight } from "react-icons/fa6";
import { FaExternalLinkAlt, FaHome } from "react-icons/fa";
import { useTranslation } from "react-i18next";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useUnifiedActiveHost } from "#/hooks/query/use-unified-active-host";
import { PathForm } from "#/components/features/served-host/path-form";
import { I18nKey } from "#/i18n/declaration";
import ServerProcessIcon from "#/icons/server-process.svg?react";

function ServedApp() {
  const { t } = useTranslation();
  const { conversationId } = useConversationId();
  const { activeHost } = useUnifiedActiveHost();
  const [refreshKey, setRefreshKey] = React.useState(0);

  const storageKey = `served-app-path:${conversationId}`;
  const savedPath = sessionStorage.getItem(storageKey);

  const [currentActiveHost, setCurrentActiveHost] = React.useState<
    string | null
  >(null);
  const [path, setPath] = React.useState<string>(savedPath ?? "hello");

  const formRef = React.useRef<HTMLFormElement>(null);

  React.useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (event.data?.type === "route-change" && event.data.path) {
        // Persist the iframe's current pathname so a refresh restores it,
        // but do NOT call setPath — setting path would re-render the iframe
        // with a new src and force-reload it, undoing the client-side nav.
        const pathname = event.data.path.split("?")[0];
        sessionStorage.setItem(storageKey, pathname);
      }
    };
    window.addEventListener("message", handleMessage);
    return () => window.removeEventListener("message", handleMessage);
  }, [storageKey]);

  const handleOnBlur = () => {
    if (formRef.current) {
      const formData = new FormData(formRef.current);
      const urlInputValue = formData.get("url")?.toString();

      if (urlInputValue) {
        const url = new URL(urlInputValue);

        setCurrentActiveHost(url.origin);
        setPath(url.pathname);
        sessionStorage.setItem(storageKey, url.pathname);
      }
    }
  };

  const resetUrl = () => {
    setCurrentActiveHost(activeHost);
    setPath("");
    sessionStorage.removeItem(storageKey);

    if (formRef.current) {
      formRef.current.reset();
    }
  };

  React.useEffect(() => {
    if (savedPath) {
      setCurrentActiveHost(activeHost);
    } else {
      resetUrl();
    }
  }, [activeHost]);

  if (!currentActiveHost) {
    return (
      <div className="flex flex-col items-center justify-center w-full h-full p-10">
        <ServerProcessIcon width={113} height={113} color="#A1A1A1" />
        <span className="text-[#8D95A9] text-[19px] font-normal leading-5">
          {t(I18nKey.BROWSER$SERVER_MESSAGE)}
        </span>
      </div>
    );
  }

  const fullUrl = (() => {
    if (!path) return currentActiveHost;
    try {
      const url = new URL(currentActiveHost);
      url.pathname = path;
      return url.toString();
    } catch {
      return currentActiveHost;
    }
  })();

  const externalUrl = (() => {
    try {
      const url = new URL(fullUrl);
      url.search = "";
      return url.toString();
    } catch {
      return fullUrl;
    }
  })();

  return (
    <div className="h-full w-full flex flex-col">
      <div className="browser-bar w-full p-2 flex items-center gap-4 border-b border-neutral-600">
        <button
          type="button"
          onClick={() => window.open(externalUrl, "_blank")}
          className="text-sm"
          aria-label={t(I18nKey.BUTTON$OPEN_IN_NEW_TAB)}
        >
          <FaExternalLinkAlt className="w-4 h-4" />
        </button>
        <button
          type="button"
          onClick={() => {
            // Refresh at the iframe's most recent pathname (persisted on
            // route-change messages), not the stale initial `path` state.
            const latest = sessionStorage.getItem(storageKey);
            if (latest) setPath(latest);
            setRefreshKey((prev) => prev + 1);
          }}
          className="text-sm"
          aria-label={t(I18nKey.BUTTON$REFRESH)}
        >
          <FaArrowRotateRight className="w-4 h-4" />
        </button>

        <button
          type="button"
          onClick={() => resetUrl()}
          className="text-sm"
          aria-label={t(I18nKey.BUTTON$HOME)}
        >
          <FaHome className="w-4 h-4" />
        </button>
        <div className="w-full flex">
          <PathForm
            ref={formRef}
            onBlur={handleOnBlur}
            defaultValue={fullUrl}
          />
        </div>
      </div>
      <iframe
        key={refreshKey}
        title={t(I18nKey.SERVED_APP$TITLE)}
        src={fullUrl}
        className="w-full h-full custom-scrollbar-always"
      />
    </div>
  );
}

export default ServedApp;
