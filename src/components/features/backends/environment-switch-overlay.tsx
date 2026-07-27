import React from "react";
import { createPortal } from "react-dom";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import {
  getEnvironmentSwitchSnapshot,
  subscribeEnvironmentSwitch,
} from "./environment-switch-store";

// Re-export the store API so existing call sites and tests that import from
// this module keep working unchanged.
export {
  ENVIRONMENT_SWITCH_DURATION_MS,
  ENVIRONMENT_SWITCH_SETACTIVE_DELAY_MS,
  triggerEnvironmentSwitch,
  dismissEnvironmentSwitch,
  __resetEnvironmentSwitchOverlayForTests,
} from "./environment-switch-store";

function EnvironmentSwitchIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 74.17 22"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden
    >
      <g>
        <rect
          x="1"
          y="1"
          width="20"
          height="8"
          rx="2"
          ry="2"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
        />
        <rect
          x="1"
          y="13"
          width="20"
          height="8"
          rx="2"
          ry="2"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
        />
        <line
          x1="5"
          y1="5"
          x2="5.01"
          y2="5"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        <line
          x1="5"
          y1="17"
          x2="5.01"
          y2="17"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
      </g>
      <g>
        <rect
          x="53.17"
          y="1"
          width="20"
          height="8"
          rx="2"
          ry="2"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
        />
        <rect
          x="53.17"
          y="13"
          width="20"
          height="8"
          rx="2"
          ry="2"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
        />
        <line
          x1="57.17"
          y1="5"
          x2="57.18"
          y2="5"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        <line
          x1="57.17"
          y1="17"
          x2="57.18"
          y2="17"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
      </g>
      <g>
        <path
          d="M43.09,7l4,4-4,4"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M27.09,11h20"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M31.09,7l-4,4,4,4"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </g>
    </svg>
  );
}

export function EnvironmentSwitchOverlay() {
  const { t } = useTranslation("openhands");
  const { visible, target } = React.useSyncExternalStore(
    subscribeEnvironmentSwitch,
    getEnvironmentSwitchSnapshot,
    getEnvironmentSwitchSnapshot,
  );

  if (!visible || typeof document === "undefined") return null;

  return createPortal(
    <div
      data-testid="environment-switch-overlay"
      data-target={target}
      className="environment-switch-overlay pointer-events-none fixed inset-0 z-[2147483646] flex items-center justify-center"
      aria-live="polite"
      aria-atomic="true"
    >
      <div className="pointer-events-none flex min-w-[280px] max-w-[420px] flex-col items-center gap-2 rounded-xl border border-[var(--oh-border)] bg-[var(--oh-surface)] px-5 py-4 text-[var(--oh-foreground)] shadow-2xl">
        <EnvironmentSwitchIcon className="mb-2 h-6 w-20 shrink-0 text-[var(--oh-foreground)]" />
        <p className="text-center text-sm font-medium">
          {t(I18nKey.BACKEND$SWITCHING_TO, { environment: target })}
        </p>
      </div>
    </div>,
    document.body,
  );
}

// React.lazy() expects a default export.
export default EnvironmentSwitchOverlay;
