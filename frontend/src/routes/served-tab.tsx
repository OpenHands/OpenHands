import React from "react";
import { FaArrowRotateRight } from "react-icons/fa6";
import { FaExternalLinkAlt, FaHome } from "react-icons/fa";
import { useTranslation } from "react-i18next";
import { useUnifiedActiveHost } from "#/hooks/query/use-unified-active-host";
import { PathForm } from "#/components/features/served-host/path-form";
import { I18nKey } from "#/i18n/declaration";
import ServerProcessIcon from "#/icons/server-process.svg?react";

function ServedApp() {
  const { t } = useTranslation();
  const { activeHost } = useUnifiedActiveHost();
  const [refreshKey, setRefreshKey] = React.useState(0);
  const [currentActiveHost, setCurrentActiveHost] = React.useState<
    string | null
  >(null);
  const [path, setPath] = React.useState<string>("hello");

  const formRef = React.useRef<HTMLFormElement>(null);

  const handleOnBlur = () => {
    if (formRef.current) {
      const formData = new FormData(formRef.current);
      const urlInputValue = formData.get("url")?.toString();

      if (urlInputValue) {
        const url = new URL(urlInputValue);

        setCurrentActiveHost(url.origin);
        setPath(url.pathname);
      }
    }
  };

  const resetUrl = () => {
    setCurrentActiveHost(activeHost);
    setPath("");

    if (formRef.current) {
      formRef.current.reset();
    }
  };

  React.useEffect(() => {
    resetUrl();
  }, [activeHost]);

  const fullUrl = `${currentActiveHost}/${path}`;

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

  return (
    <div className="h-full w-full">
      <div className="w-full p-2 flex items-center gap-4 border-b border-neutral-600">
        <button
          type="button"
          onClick={() => window.open(fullUrl, "_blank")}
          className="text-sm"
          aria-label={t(I18nKey.BUTTON$OPEN_IN_NEW_TAB)}
        >
          <FaExternalLinkAlt className="w-4 h-4" />
        </button>
        <button
          type="button"
          onClick={() => setRefreshKey((prev) => prev + 1)}
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
