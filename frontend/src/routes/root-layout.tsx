import React from "react";
import {
  useRouteError,
  isRouteErrorResponse,
  Outlet,
  useNavigate,
  useLocation,
  useSearchParams,
} from "react-router";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import i18n from "#/i18n";
import { useIsAuthed } from "#/hooks/query/use-is-authed";
import { useConfig } from "#/hooks/query/use-config";
import { Sidebar } from "#/components/features/sidebar/sidebar";
import { ReauthModal } from "#/components/features/waitlist/reauth-modal";
import { AnalyticsConsentFormModal } from "#/components/features/analytics/analytics-consent-form-modal";
import { useSettings } from "#/hooks/query/use-settings";
import { useMigrateUserConsent } from "#/hooks/use-migrate-user-consent";
import { SetupPaymentModal } from "#/components/features/payment/setup-payment-modal";
import { displaySuccessToast } from "#/utils/custom-toast-handlers";
import { useIsOnIntermediatePage } from "#/hooks/use-is-on-intermediate-page";
import { useAutoLogin } from "#/hooks/use-auto-login";
import { useAuthCallback } from "#/hooks/use-auth-callback";
import { useReoTracking } from "#/hooks/use-reo-tracking";
import { useSyncPostHogConsent } from "#/hooks/use-sync-posthog-consent";
import { LOCAL_STORAGE_KEYS } from "#/utils/local-storage";
import { EmailVerificationGuard } from "#/components/features/guards/email-verification-guard";
import { AlertBanner } from "#/components/features/alerts/alert-banner";
import { cn, isMobileDevice } from "#/utils/utils";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { useAppTitle } from "#/hooks/use-app-title";

export function ErrorBoundary() {
  const error = useRouteError();
  const { t } = useTranslation();

  if (isRouteErrorResponse(error)) {
    return (
      <div>
        <h1>{error.status}</h1>
        <p>{error.statusText}</p>
        <pre>
          {error.data instanceof Object
            ? JSON.stringify(error.data)
            : error.data}
        </pre>
      </div>
    );
  }
  if (error instanceof Error) {
    return (
      <div>
        <h1>{t(I18nKey.ERROR$GENERIC)}</h1>
        <pre>{error.message}</pre>
      </div>
    );
  }

  return (
    <div>
      <h1>{t(I18nKey.ERROR$UNKNOWN)}</h1>
    </div>
  );
}

export default function MainApp() {
  const appTitle = useAppTitle();
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const [searchParams] = useSearchParams();
  const isOnIntermediatePage = useIsOnIntermediatePage();
  const { data: settings } = useSettings();
  const { migrateUserConsent } = useMigrateUserConsent();
  const { t } = useTranslation();

  const config = useConfig();
  const {
    data: isAuthed,
    isFetching: isFetchingAuth,
    isLoading: isAuthLoading,
    isError: isAuthError,
  } = useIsAuthed();

  const [consentFormIsOpen, setConsentFormIsOpen] = React.useState(false);

  // Auto-login if login method is stored in local storage
  useAutoLogin();

  // Handle authentication callback and set login method after successful authentication
  useAuthCallback();

  // Initialize Reo.dev tracking in SaaS mode
  useReoTracking();

  // Sync PostHog opt-in/out state with backend setting on mount
  useSyncPostHogConsent();

  React.useEffect(() => {
    // Don't change language when on intermediate pages (TOS, profile questions)
    if (!isOnIntermediatePage && settings?.language) {
      i18n.changeLanguage(settings.language);
    }
  }, [settings?.language, isOnIntermediatePage]);

  React.useEffect(() => {
    // Don't show consent form when on intermediate pages
    if (!isOnIntermediatePage) {
      const consentFormModalIsOpen =
        settings?.user_consents_to_analytics === null;

      setConsentFormIsOpen(consentFormModalIsOpen);
    }
  }, [settings, isOnIntermediatePage]);

  React.useEffect(() => {
    // Don't migrate user consent when on intermediate pages
    if (!isOnIntermediatePage) {
      // Migrate user consent to the server if it was previously stored in localStorage
      migrateUserConsent({
        handleAnalyticsWasPresentInLocalStorage: () => {
          setConsentFormIsOpen(false);
        },
      });
    }
  }, [isOnIntermediatePage]);

  React.useEffect(() => {
    if (settings?.is_new_user && config.data?.app_mode === "saas") {
      displaySuccessToast(t(I18nKey.BILLING$YOURE_IN));
    }
  }, [settings?.is_new_user, config.data?.app_mode]);

  // Function to check if login method exists in local storage
  const checkLoginMethodExists = React.useCallback(() => {
    // Only check localStorage if we're in a browser environment
    if (typeof window !== "undefined" && window.localStorage) {
      return localStorage.getItem(LOCAL_STORAGE_KEYS.LOGIN_METHOD) !== null;
    }
    return false;
  }, []);

  // State to track if login method exists
  const [loginMethodExists, setLoginMethodExists] = React.useState(
    checkLoginMethodExists(),
  );

  // Listen for storage events to update loginMethodExists when logout happens
  React.useEffect(() => {
    const handleStorageChange = (event: StorageEvent) => {
      if (event.key === LOCAL_STORAGE_KEYS.LOGIN_METHOD) {
        setLoginMethodExists(checkLoginMethodExists());
      }
    };

    // Also check on window focus, as logout might happen in another tab
    const handleWindowFocus = () => {
      setLoginMethodExists(checkLoginMethodExists());
    };

    window.addEventListener("storage", handleStorageChange);
    window.addEventListener("focus", handleWindowFocus);

    return () => {
      window.removeEventListener("storage", handleStorageChange);
      window.removeEventListener("focus", handleWindowFocus);
    };
  }, [checkLoginMethodExists]);

  // Check login method status when auth status changes
  React.useEffect(() => {
    // When auth status changes (especially on logout), recheck login method
    setLoginMethodExists(checkLoginMethodExists());
  }, [isAuthed, checkLoginMethodExists]);

  const shouldRedirectToLogin =
    config.isLoading ||
    isAuthLoading ||
    (!isAuthed &&
      !isAuthError &&
      !isOnIntermediatePage &&
      config.data?.app_mode === "saas" &&
      !loginMethodExists);

  React.useEffect(() => {
    if (shouldRedirectToLogin) {
      // Include search params in returnTo to preserve query string (e.g., user_code for device OAuth)
      const searchString = searchParams.toString();
      let fullPath = "";
      if (pathname !== "/") {
        fullPath = searchString ? `${pathname}?${searchString}` : pathname;
      }
      const loginUrl = fullPath
        ? `/login?returnTo=${encodeURIComponent(fullPath)}`
        : "/login";
      navigate(loginUrl, { replace: true });
    }
  }, [shouldRedirectToLogin, pathname, searchParams, navigate]);

  if (shouldRedirectToLogin) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-base">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  const renderReAuthModal =
    !isAuthed &&
    !isAuthError &&
    !isFetchingAuth &&
    !isOnIntermediatePage &&
    config.data?.app_mode === "saas" &&
    loginMethodExists;

  return (
    <div
      data-testid="root-layout"
      className={cn(
        "h-screen lg:min-w-5xl flex flex-col md:flex-row bg-base",
        pathname === "/" ? "p-0" : "p-0 md:p-3 md:pl-0",
        isMobileDevice() && "overflow-hidden",
      )}
    >
      <title>{appTitle}</title>
      <Sidebar />

      <div className="flex flex-col w-full h-[calc(100%-50px)] md:h-full gap-3">
        {config.data &&
          (config.data.maintenance_start_time ||
            (config.data.faulty_models &&
              config.data.faulty_models.length > 0) ||
            config.data.error_message) && (
            <AlertBanner
              maintenanceStartTime={config.data.maintenance_start_time}
              faultyModels={config.data.faulty_models}
              errorMessage={config.data.error_message}
              updatedAt={config.data.updated_at}
            />
          )}
        <div
          id="root-outlet"
          className="flex-1 relative overflow-auto custom-scrollbar"
        >
          <EmailVerificationGuard>
            <Outlet />
          </EmailVerificationGuard>
        </div>
      </div>

      {renderReAuthModal && <ReauthModal />}
      {config.data?.app_mode === "oss" && consentFormIsOpen && (
        <AnalyticsConsentFormModal
          onClose={() => {
            setConsentFormIsOpen(false);
          }}
        />
      )}

      {config.data?.feature_flags.enable_billing &&
        config.data?.app_mode === "saas" &&
        settings?.is_new_user && <SetupPaymentModal />}
    </div>
  );
}
