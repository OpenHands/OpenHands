import React, { CSSProperties, ReactNode } from "react";
import { CircleX } from "lucide-react";
import toast, { ToastOptions } from "react-hot-toast";
import { OH_STATUS_ERROR_COLOR } from "#/constants/status-colors";
import { calculateToastDuration } from "./toast-duration";
import { cn } from "#/utils/utils";
import i18n from "#/i18n";
import { I18nKey } from "#/i18n/declaration";
import {
  isBackendRequestTimeoutMessage,
  isCorsOrNetworkErrorMessage,
} from "./user-facing-error";

// react-hot-toast accepts only CSSProperties via the style option — cannot use className
const TOAST_STYLE: CSSProperties = {
  background: "var(--oh-color-tertiary)",
  border: "1px solid var(--oh-border-input)",
  color: "#fff",
  borderRadius: "var(--oh-radius)",
  maxWidth: "400px",
  wordBreak: "break-word",
  overflowWrap: "anywhere",
  whiteSpace: "pre-wrap",
};

export const TOAST_OPTIONS: ToastOptions = {
  position: "top-right",
  style: TOAST_STYLE,
};

const ERROR_TOAST_STYLE: CSSProperties = {
  ...TOAST_STYLE,
  color: "var(--oh-muted)",
};

/** Icon + message row; center icon for single-line text, top-align when wrapped. */
export function ErrorToastContent({ message }: { message: ReactNode }) {
  const contentRef = React.useRef<HTMLSpanElement>(null);
  const [isMultiLine, setIsMultiLine] = React.useState(false);

  React.useLayoutEffect(() => {
    const content = contentRef.current;
    if (!content) {
      return undefined;
    }

    const updateIsMultiLine = () => {
      const lineHeight = Number.parseFloat(
        getComputedStyle(content).lineHeight,
      );
      if (!Number.isFinite(lineHeight) || lineHeight <= 0) {
        setIsMultiLine(false);
        return;
      }

      setIsMultiLine(content.getBoundingClientRect().height > lineHeight * 1.5);
    };

    updateIsMultiLine();

    const observer = new ResizeObserver(updateIsMultiLine);
    observer.observe(content);

    return () => observer.disconnect();
  }, [message]);

  return (
    <div
      className={cn(
        "flex min-w-0 gap-2",
        isMultiLine ? "items-start" : "items-center",
      )}
    >
      <CircleX
        aria-hidden
        className="h-4 w-4 shrink-0"
        strokeWidth={2}
        style={{ color: OH_STATUS_ERROR_COLOR }}
      />
      <span
        ref={contentRef}
        className="min-w-0 flex-1 text-sm leading-5 [word-break:break-word] [overflow-wrap:anywhere]"
      >
        {message}
      </span>
    </div>
  );
}

export const ERROR_TOAST_OPTIONS: ToastOptions = {
  ...TOAST_OPTIONS,
  icon: null,
  style: ERROR_TOAST_STYLE,
};

export const displayErrorToast = (error: string | null | undefined) => {
  let errorMessage = error || i18n.t(I18nKey.STATUS$ERROR);
  if (isCorsOrNetworkErrorMessage(errorMessage)) {
    errorMessage = i18n.t(I18nKey.ERROR$CORS_OR_NETWORK);
  } else if (isBackendRequestTimeoutMessage(errorMessage)) {
    errorMessage = i18n.t(I18nKey.ERROR$BACKEND_REQUEST_TIMEOUT);
  }
  const duration = calculateToastDuration(errorMessage, 4000);
  toast(<ErrorToastContent message={errorMessage} />, {
    ...ERROR_TOAST_OPTIONS,
    duration,
  });
};

export const displaySuccessToast = (message: string) => {
  const duration = calculateToastDuration(message, 5000);
  toast.success(
    <span className="[word-break:break-word] [overflow-wrap:anywhere]">
      {message}
    </span>,
    { ...TOAST_OPTIONS, duration },
  );
};

export const displaySuccessToastWithLink = (
  message: string,
  linkLabel: string,
  href: string,
) => {
  const duration = calculateToastDuration(`${message} ${linkLabel}`, 5000);
  toast.success(
    <span className="[word-break:break-word] [overflow-wrap:anywhere]">
      {message}{" "}
      <a className="underline hover:no-underline" href={href}>
        {linkLabel}
      </a>
    </span>,
    { ...TOAST_OPTIONS, duration },
  );
};

/**
 * Neutral, non-success notice — used when an action completed but the outcome
 * is qualified (e.g. a secret was saved but the active backend can't consume it
 * yet). Renders without the success checkmark so it doesn't read as "all good".
 */
export const displayWarningToast = (message: string) => {
  const duration = calculateToastDuration(message, 6000);
  toast(
    <span className="[word-break:break-word] [overflow-wrap:anywhere]">
      {message}
    </span>,
    { ...TOAST_OPTIONS, icon: "⚠️", duration },
  );
};
