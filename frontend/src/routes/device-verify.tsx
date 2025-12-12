/* eslint-disable i18next/no-literal-string */
import React, { useEffect, useState } from "react";
import { useSearchParams } from "react-router";
import { useIsAuthed } from "#/hooks/query/use-is-authed";

export function DeviceVerify() {
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

  useEffect(() => {
    // If user is authenticated and we have a user_code, process verification
    if (isAuthed && userCode && !verificationResult && !isProcessing) {
      processDeviceVerification(userCode);
    }
  }, [isAuthed, userCode, verificationResult, isProcessing]);

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

  // Show manual code entry form if no code in URL or user is authenticated
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
