/* eslint-disable i18next/no-literal-string */
import React, { useState } from "react";
import { useSearchParams } from "react-router";
import { useIsAuthed } from "#/hooks/query/use-is-authed";

export default function DeviceVerify() {
  const [searchParams] = useSearchParams();
  const { data: isAuthed, isLoading: isAuthLoading } = useIsAuthed();
  const [verificationResult, setVerificationResult] = useState<{
    success: boolean;
    message: string;
  } | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

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
          message:
            "Device authorized successfully! You can now return to your CLI and close this window.",
        });
      } else {
        const errorText = await response.text();
        setVerificationResult({
          success: false,
          message: errorText || "Failed to authorize device. Please try again.",
        });
      }
    } catch (error) {
      setVerificationResult({
        success: false,
        message:
          "An error occurred while authorizing the device. Please try again.",
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
              {verificationResult.success ? "Success!" : "Error"}
            </h2>
            <p className="text-muted-foreground mb-4">
              {verificationResult.message}
            </p>
            {!verificationResult.success && (
              <button
                type="button"
                onClick={() => window.location.reload()}
                className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
              >
                Try Again
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
              Processing device verification...
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Show device authorization confirmation if user is authenticated and code is provided
  if (isAuthed && userCode) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="max-w-md w-full mx-auto p-6 bg-card rounded-lg shadow-lg">
          <h1 className="text-2xl font-bold mb-4 text-center">
            Device Authorization Request
          </h1>
          <div className="mb-6 p-4 bg-muted rounded-lg">
            <p className="text-sm text-muted-foreground mb-2">Device Code:</p>
            <p className="text-lg font-mono font-semibold text-center tracking-wider">
              {userCode}
            </p>
          </div>
          <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-start">
              <svg
                className="w-5 h-5 text-yellow-600 mt-0.5 mr-2 flex-shrink-0"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16c-.77.833.192 2.5 1.732 2.5z"
                />
              </svg>
              <div>
                <p className="text-sm font-medium text-yellow-800 mb-1">
                  Security Notice
                </p>
                <p className="text-sm text-yellow-700">
                  Only authorize this device if you initiated this request from
                  your CLI or application.
                </p>
              </div>
            </div>
          </div>
          <p className="text-muted-foreground mb-6 text-center">
            Do you want to authorize this device to access your OpenHands
            account?
          </p>
          <div className="flex gap-3">
            <button
              type="button"
              onClick={() => window.close()}
              className="flex-1 px-4 py-2 border border-input rounded-md hover:bg-muted"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => processDeviceVerification(userCode)}
              className="flex-1 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
            >
              Authorize Device
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Show manual code entry form if no code in URL but user is authenticated
  if (isAuthed && !userCode) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="max-w-md w-full mx-auto p-6 bg-card rounded-lg shadow-lg">
          <h1 className="text-2xl font-bold mb-4 text-center">
            Device Authorization
          </h1>
          <p className="text-muted-foreground mb-6 text-center">
            Enter the code displayed on your device:
          </p>
          <form onSubmit={handleManualSubmit}>
            <div className="mb-4">
              <label
                htmlFor="user_code"
                className="block text-sm font-medium mb-2"
              >
                Device Code:
              </label>
              <input
                type="text"
                id="user_code"
                name="user_code"
                required
                className="w-full px-3 py-2 border border-input rounded-md focus:outline-none focus:ring-2 focus:ring-ring"
                placeholder="Enter your device code"
              />
            </div>
            <button
              type="submit"
              className="w-full px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
            >
              Continue
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
            Processing device verification...
          </p>
        </div>
      </div>
    );
  }

  // Show authentication required message (this will trigger the auth modal via root layout)
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="max-w-md w-full mx-auto p-6 bg-card rounded-lg shadow-lg text-center">
        <h1 className="text-2xl font-bold mb-4">Authentication Required</h1>
        <p className="text-muted-foreground">
          Please sign in to authorize your device.
        </p>
      </div>
    </div>
  );
}
