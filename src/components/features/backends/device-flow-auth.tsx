import React from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { useDeviceFlow, type DeviceFlowStatus } from "#/hooks/use-device-flow";
import type { CloudConnectionSource } from "#/services/cloud-funnel-analytics";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

type DeviceFlowButtonVariant =
  | "primary"
  | "secondary"
  | "tertiary"
  | "unstyled";
type DeviceFlowStatusDisplay = "inline" | "modal";

interface DeviceFlowAuthProps {
  /** The host URL for the cloud backend */
  host: string;
  /** Callback when authentication succeeds with the API key */
  onSuccess: (apiKey: string) => void;
  /** Test ID prefix for the component */
  testIdRoot: string;
  /** Whether the login button should be disabled (e.g., when no host is entered) */
  isDisabled?: boolean;
  /** Override for the idle button label and icon-only accessible name. */
  idleButtonLabel?: string;
  /** Optional visible content for the idle button. Defaults to the idle label. */
  idleButtonContent?: React.ReactNode;
  /** Optional classes for the root wrapper. */
  className?: string;
  /** Optional classes for the idle button. */
  buttonClassName?: string;
  /** Visual variant for the idle button. */
  buttonVariant?: DeviceFlowButtonVariant;
  /** Whether in-progress auth content should render inline or in a modal. */
  statusDisplay?: DeviceFlowStatusDisplay;
  /** Product surface that initiated this authorization attempt. */
  analyticsSource?: CloudConnectionSource;
}

/**
 * Device Flow authentication UI component.
 *
 * Shows a "Login with OpenHands Cloud" button that initiates OAuth 2.0 Device Flow
 * authentication. Displays status during the auth process and auto-opens
 * the browser for user authorization.
 */
/**
 * Validate that a URL is safe to open in a popup.
 * Prevents XSS via javascript: URLs or other malicious schemes.
 */
function isValidVerificationUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:";
  } catch {
    return false;
  }
}

export function DeviceFlowAuth({
  host,
  onSuccess,
  testIdRoot,
  isDisabled = false,
  idleButtonLabel,
  idleButtonContent,
  className,
  buttonClassName,
  buttonVariant = "primary",
  statusDisplay = "inline",
  analyticsSource,
}: DeviceFlowAuthProps) {
  const { t } = useTranslation("openhands");
  const deviceFlow = useDeviceFlow();
  const popupRef = React.useRef<Window | null>(null);

  // Close popup on unmount or when auth completes/errors
  React.useEffect(() => {
    return () => {
      popupRef.current?.close();
    };
  }, []);

  // Update popup URL when verification URL becomes available
  React.useEffect(() => {
    if (
      deviceFlow.status === "awaiting_authorization" &&
      deviceFlow.verificationUrl &&
      popupRef.current &&
      !popupRef.current.closed
    ) {
      // Validate URL before assigning to prevent XSS
      if (!isValidVerificationUrl(deviceFlow.verificationUrl)) {
        console.error("Invalid verification URL protocol");
        return;
      }
      try {
        popupRef.current.location.href = deviceFlow.verificationUrl;
      } catch {
        // Cross-origin error - popup was navigated away
        // Open a new one as fallback
        popupRef.current = window.open(
          deviceFlow.verificationUrl,
          "_blank",
          "noopener,noreferrer",
        );
      }
    }
  }, [deviceFlow.status, deviceFlow.verificationUrl]);

  // Call onSuccess when authentication completes
  React.useEffect(() => {
    if (deviceFlow.status === "success" && deviceFlow.apiKey) {
      try {
        onSuccess(deviceFlow.apiKey);
      } finally {
        deviceFlow.reset();
        popupRef.current?.close();
      }
    }
  }, [deviceFlow.status, deviceFlow.apiKey, deviceFlow.reset, onSuccess]);

  const handleStartAuth = () => {
    // Normalize and validate the host URL
    const normalizedHost = host.trim().replace(/\/+$/, "");
    const fullHost = /^https?:\/\//i.test(normalizedHost)
      ? normalizedHost
      : `https://${normalizedHost}`;

    // Validate URL before proceeding
    try {
      const url = new URL(fullHost);
      // Check for URL manipulation attacks
      if (url.username || url.password) {
        throw new Error("Invalid URL format");
      }
    } catch {
      return; // Invalid URL, don't proceed
    }

    // Open popup immediately on user click to avoid popup blocker
    // Start with about:blank and update URL once we have verification URL
    // Note: We intentionally don't use "noopener" here because we need to
    // maintain a reference to update the popup's location when the
    // verification URL becomes available
    popupRef.current = window.open("about:blank", "_blank");

    if (!popupRef.current) {
      // Popup was blocked - flow will still work, user can click manual link
      console.warn("Popup blocked - user will need to use manual link");
    }

    deviceFlow.start(fullHost, analyticsSource);
  };

  const handleCancel = () => {
    deviceFlow.cancel();
    popupRef.current?.close();
  };

  const statusContent = (
    <DeviceFlowStatusContent
      status={deviceFlow.status}
      error={deviceFlow.error}
      verificationUrl={deviceFlow.verificationUrl}
      testIdRoot={testIdRoot}
      onCancel={handleCancel}
      onRetry={handleStartAuth}
    />
  );
  const showStatusModal =
    statusDisplay === "modal" && deviceFlow.status !== "idle";
  const idleLabel = idleButtonLabel ?? t(I18nKey.BACKEND$LOGIN_WITH_OPENHANDS);

  return (
    <div
      data-testid={`${testIdRoot}-device-flow`}
      className={cn("flex flex-col gap-3", className)}
    >
      {deviceFlow.status === "idle" && buttonVariant === "unstyled" && (
        <button
          type="button"
          onClick={handleStartAuth}
          data-testid={`${testIdRoot}-login-button`}
          className={buttonClassName}
          disabled={isDisabled}
          aria-label={idleButtonContent ? idleLabel : undefined}
        >
          {idleButtonContent ?? idleLabel}
        </button>
      )}

      {deviceFlow.status === "idle" && buttonVariant !== "unstyled" && (
        <BrandButton
          type="button"
          variant={buttonVariant}
          onClick={handleStartAuth}
          testId={`${testIdRoot}-login-button`}
          className={cn("w-full", buttonClassName)}
          isDisabled={isDisabled}
          ariaLabel={idleButtonContent ? idleLabel : undefined}
        >
          {idleButtonContent ?? idleLabel}
        </BrandButton>
      )}

      {statusDisplay === "inline" ? statusContent : null}

      {showStatusModal ? (
        <ModalBackdrop
          onClose={handleCancel}
          aria-label={t(I18nKey.BACKEND$LOGIN_WITH_OPENHANDS)}
          closeOnBackdropClick={false}
        >
          <ModalBody
            testID={`${testIdRoot}-auth-modal`}
            width="sm"
            className="items-stretch border border-[var(--oh-border)]"
          >
            {statusContent}
          </ModalBody>
        </ModalBackdrop>
      ) : null}
    </div>
  );
}

interface DeviceFlowStatusContentProps {
  status: DeviceFlowStatus;
  error: string | null;
  verificationUrl: string | null;
  testIdRoot: string;
  onCancel: () => void;
  onRetry: () => void;
}

function DeviceFlowStatusContent({
  status,
  error,
  verificationUrl,
  testIdRoot,
  onCancel,
  onRetry,
}: DeviceFlowStatusContentProps) {
  const { t } = useTranslation("openhands");

  if (status === "idle" || status === "success") return null;

  if (status === "starting") {
    return (
      <div
        className="flex items-center gap-2 rounded-lg bg-base-tertiary p-3"
        data-testid={`${testIdRoot}-auth-starting`}
        role="status"
        aria-live="polite"
      >
        <LoadingSpinner />
        <span className="text-sm text-[var(--oh-text-tertiary)]">
          {t(I18nKey.BACKEND$AUTH_STARTING)}
        </span>
      </div>
    );
  }

  if (status === "awaiting_authorization") {
    const validVerificationUrl =
      verificationUrl && isValidVerificationUrl(verificationUrl)
        ? verificationUrl
        : null;

    return (
      <div
        className="flex flex-col gap-4 rounded-lg bg-base-tertiary p-4"
        data-testid={`${testIdRoot}-auth-awaiting`}
        role="status"
        aria-live="polite"
      >
        <div className="flex items-center gap-2">
          <LoadingSpinner />
          <span className="text-sm font-medium text-white">
            {t(I18nKey.BACKEND$AUTH_AWAITING)}
          </span>
        </div>
        <p className="text-sm leading-5 text-[var(--oh-text-tertiary)]">
          {t(I18nKey.BACKEND$AUTH_BROWSER_OPENED)}
        </p>
        {validVerificationUrl ? (
          <div className="flex flex-col gap-1 rounded-md border border-[var(--oh-border)] bg-[var(--oh-surface)] p-3 text-xs text-[var(--oh-muted)]">
            <p>{t(I18nKey.BACKEND$AUTH_OPEN_MANUALLY)}</p>
            <a
              href={validVerificationUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="max-h-16 overflow-auto break-all text-blue-400 hover:underline"
            >
              {validVerificationUrl}
            </a>
          </div>
        ) : null}
        <BrandButton
          type="button"
          variant="secondary"
          onClick={onCancel}
          testId={`${testIdRoot}-auth-cancel`}
          className="w-full"
        >
          {t(I18nKey.BACKEND$AUTH_CANCEL)}
        </BrandButton>
      </div>
    );
  }

  return (
    <div
      className="flex flex-col gap-3 rounded-lg border border-red-700 bg-red-900/20 p-4"
      data-testid={`${testIdRoot}-auth-error`}
      role="alert"
    >
      <p className="text-sm text-red-400">{error}</p>
      <BrandButton
        type="button"
        variant="secondary"
        onClick={onRetry}
        testId={`${testIdRoot}-auth-retry`}
        className="w-full"
      >
        {t(I18nKey.BACKEND$AUTH_RETRY)}
      </BrandButton>
    </div>
  );
}

function LoadingSpinner() {
  return (
    <svg
      className="animate-spin h-4 w-4 text-white"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}
