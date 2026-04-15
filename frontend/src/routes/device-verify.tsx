import React, { useState } from "react";
import { useSearchParams } from "react-router";
import { useTranslation } from "react-i18next";
import { useIsAuthed } from "#/hooks/query/use-is-authed";
import { LoginCTA } from "#/components/features/auth/login-cta";
import { I18nKey } from "#/i18n/declaration";
import { H1 } from "#/ui/typography";
import { ENABLE_PROJ_USER_JOURNEY } from "#/utils/feature-flags";

export default function DeviceVerify() {
  const { t } = useTranslation();
  const [searchParams] = useSearchParams();
  const { data: isAuthed, isLoading: isAuthLoading } = useIsAuthed();
  const [verificationResult, setVerificationResult] = useState<{
    success: boolean;
    messageKey: I18nKey;
  } | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const showEnterpriseBanner = ENABLE_PROJ_USER_JOURNEY();

  // Get user_code from URL parameters
  const userCode = searchParams.get("user_code");

  const processDeviceVerification = async (code: string) => {
    try {
      setIsProcessing(true);

      // Call the backend API endpoint to process device verification
      const response = await fetch("/oauth/device/verify-authenticated", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: `user_code=${encodeURIComponent(code)}`,
        credentials: "include", // Include cookies for authentication
      });

      if (response.ok) {
        // Show success message
        setVerificationResult({
          success: true,
          messageKey: I18nKey.DEVICE$SUCCESS_MESSAGE,
        });
      } else {
        setVerificationResult({
          success: false,
          messageKey: I18nKey.DEVICE$ERROR_FAILED,
        });
      }
    } catch (error) {
      setVerificationResult({
        success: false,
        messageKey: I18nKey.DEVICE$ERROR_OCCURRED,
      });
    } finally {
      setIsProcessing(false);
    }
  };

  // Remove automatic verification - require explicit user consent

  const handleManualSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const code = formData.get("user_code") as string;
    if (code && isAuthed) {
      processDeviceVerification(code);
    }
  };

  // Show verification result if we have one
  if (verificationResult) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="max-w-md w-full mx-auto p-6 bg-card rounded-lg shadow-lg">
          <div className="text-center">
            <div
              className={`mb-4 ${verificationResult.success ? "text-green-600" : "text-red-600"}`}
            >
              {verificationResult.success ? (
                <svg
                  className="w-12 h-12 mx-auto"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M5 13l4 4L19 7"
                  />
                </svg>
              ) : (
                <svg
                  className="w-12 h-12 mx-auto"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              )}
            </div>
            <h2 className="text-xl font-semibold mb-2">
              {verificationResult.success
                ? t(I18nKey.DEVICE$SUCCESS_TITLE)
                : t(I18nKey.DEVICE$ERROR_TITLE)}
            </h2>
            <p className="text-muted-foreground mb-4">
              {t(verificationResult.messageKey)}
            </p>
            {!verificationResult.success && (
              <button
                type="button"
                onClick={() => window.location.reload()}
                className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
              >
                {t(I18nKey.DEVICE$TRY_AGAIN)}
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Show processing state
  if (isProcessing) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="max-w-md w-full mx-auto p-6 bg-card rounded-lg shadow-lg">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4" />
            <p className="text-muted-foreground">
              {t(I18nKey.DEVICE$PROCESSING)}
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Show device authorization confirmation if user is authenticated and code is provided
  if (isAuthed && userCode) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background p-4">
        <div
          className={`flex flex-col lg:flex-row items-center lg:items-stretch gap-6 w-full ${showEnterpriseBanner ? "max-w-4xl" : "max-w-md"}`}
        >
          {/* Device Authorization Card */}
          <div
            className={`flex-1 min-w-0 max-w-md w-full mx-auto p-6 bg-card rounded-2xl shadow-lg border border-[#242424] ${showEnterpriseBanner ? "lg:mx-0 lg:self-stretch" : ""}`}
          >
            <H1 className="text-2xl mb-4 text-center">
              {t(I18nKey.DEVICE$AUTHORIZATION_REQUEST)}
            </H1>
            <div className="mb-6 p-4 bg-neutral-900 rounded-lg border border-neutral-700">
              <p className="text-xs text-neutral-500 mb-2 text-center uppercase tracking-wider">
                {t(I18nKey.DEVICE$CODE_LABEL)}
              </p>
              <p className="text-xl font-mono font-semibold text-center tracking-[0.3em]">
                {userCode}
              </p>
            </div>
            <div className="mb-6 p-4 bg-amber-950/50 border-l-2 border-amber-500 rounded-r-lg">
              <p className="text-sm font-medium text-amber-500 mb-1">
                {t(I18nKey.DEVICE$SECURITY_NOTICE)}
              </p>
              <p className="text-sm text-gray-400">
                {t(I18nKey.DEVICE$SECURITY_WARNING)}
              </p>
            </div>
            <p className="text-muted-foreground mb-6 text-center">
              {t(I18nKey.DEVICE$CONFIRM_PROMPT)}
            </p>
            <div className="flex gap-3">
              <button
                type="button"
                onClick={() => window.close()}
                className="flex-1 px-4 py-2 border border-neutral-600 rounded-md hover:bg-muted text-gray-300"
              >
                {t(I18nKey.DEVICE$CANCEL)}
              </button>
              <button
                type="button"
                onClick={() => processDeviceVerification(userCode)}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                {t(I18nKey.DEVICE$AUTHORIZE)}
              </button>
            </div>
          </div>

          {/* Login CTA */}
          {showEnterpriseBanner && (
            <LoginCTA source="device_verify" className="lg:self-stretch" />
          )}
        </div>
      </div>
    );
  }

  // Show manual code entry form if no code in URL but user is authenticated
  if (isAuthed && !userCode) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="max-w-md w-full mx-auto p-6 bg-card rounded-lg shadow-lg">
          <H1 className="text-2xl mb-4 text-center">
            {t(I18nKey.DEVICE$AUTHORIZATION_TITLE)}
          </H1>
          <p className="text-muted-foreground mb-6 text-center">
            {t(I18nKey.DEVICE$ENTER_CODE_PROMPT)}
          </p>
          <form onSubmit={handleManualSubmit}>
            <div className="mb-4">
              <label
                htmlFor="user_code"
                className="block text-sm font-medium mb-2"
              >
                {t(I18nKey.DEVICE$CODE_INPUT_LABEL)}
              </label>
              <input
                type="text"
                id="user_code"
                name="user_code"
                required
                className="w-full px-3 py-2 border border-input rounded-md focus:outline-none focus:ring-2 focus:ring-ring"
                placeholder={t(I18nKey.DEVICE$CODE_PLACEHOLDER)}
              />
            </div>
            <button
              type="submit"
              className="w-full px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
            >
              {t(I18nKey.DEVICE$CONTINUE)}
            </button>
          </form>
        </div>
      </div>
    );
  }

  // Show loading state while checking authentication
  if (isAuthLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4" />
          <p className="text-muted-foreground">
            {t(I18nKey.DEVICE$PROCESSING)}
          </p>
        </div>
      </div>
    );
  }

  // Show authentication required message (this will trigger the auth modal via root layout)
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="max-w-md w-full mx-auto p-6 bg-card rounded-lg shadow-lg text-center">
        <H1 className="text-2xl mb-4">{t(I18nKey.DEVICE$AUTH_REQUIRED)}</H1>
        <p className="text-muted-foreground">
          {t(I18nKey.DEVICE$SIGN_IN_PROMPT)}
        </p>
      </div>
    </div>
  );
}
