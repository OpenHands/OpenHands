import React from "react";
import {
  useRouteError,
  isRouteErrorResponse,
  Outlet,
  useNavigate,
  useLocation,
} from "react-router";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import i18n from "#/i18n";
import { useGitHubAuthUrl } from "#/hooks/use-github-auth-url";
import { useIsAuthed } from "#/hooks/query/use-is-authed";
import { useConfig } from "#/hooks/query/use-config";
import { Sidebar } from "#/components/features/sidebar/sidebar";
import { AuthModal } from "#/components/features/waitlist/auth-modal";
import { ReauthModal } from "#/components/features/waitlist/reauth-modal";
import { AnalyticsConsentFormModal } from "#/components/features/analytics/analytics-consent-form-modal";
import { useSettings } from "#/hooks/query/use-settings";
import { useMigrateUserConsent } from "#/hooks/use-migrate-user-consent";
import { useBalance } from "#/hooks/query/use-balance";
import { SetupPaymentModal } from "#/components/features/payment/setup-payment-modal";
import { displaySuccessToast } from "#/utils/custom-toast-handlers";
import { useIsOnTosPage } from "#/hooks/use-is-on-tos-page";
import { useAutoLogin } from "#/hooks/use-auto-login";
import { useAuthCallback } from "#/hooks/use-auth-callback";
import { useReoTracking } from "#/hooks/use-reo-tracking";
import { useSyncPostHogConsent } from "#/hooks/use-sync-posthog-consent";
import { LOCAL_STORAGE_KEYS } from "#/utils/local-storage";
import { EmailVerificationGuard } from "#/components/features/guards/email-verification-guard";
import { MaintenanceBanner } from "#/components/features/maintenance/maintenance-banner";
import { cn, isMobileDevice } from "#/utils/utils";

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
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const isOnTosPage = useIsOnTosPage();
  const { data: settings } = useSettings();
  const { error } = useBalance();
  const { migrateUserConsent } = useMigrateUserConsent();
  const { t } = useTranslation();

  const config = useConfig();
  const {
    data: isAuthed,
    isFetching: isFetchingAuth,
    isError: isAuthError,
  } = useIsAuthed();

  // Always call the hook, but we'll only use the result when not on TOS page
  const gitHubAuthUrl = useGitHubAuthUrl({
    appMode: config.data?.APP_MODE || null,
    gitHubClientId: config.data?.GITHUB_CLIENT_ID || null,
    authUrl: config.data?.AUTH_URL,
  });

  // When on TOS page, we don't use the GitHub auth URL
  const effectiveGitHubAuthUrl = isOnTosPage ? null : gitHubAuthUrl;

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
    // Don't change language when on TOS page
    if (!isOnTosPage && settings?.language) {
      i18n.changeLanguage(settings.language);
    }
  }, [settings?.language, isOnTosPage]);

  React.useEffect(() => {
    // Don't show consent form when on TOS page
    if (!isOnTosPage) {
      const consentFormModalIsOpen =
        settings?.user_consents_to_analytics === null;

      setConsentFormIsOpen(consentFormModalIsOpen);
    }
  }, [settings, isOnTosPage]);

  React.useEffect(() => {
    // Don't migrate user consent when on TOS page
    if (!isOnTosPage) {
      // Migrate user consent to the server if it was previously stored in localStorage
      migrateUserConsent({
        handleAnalyticsWasPresentInLocalStorage: () => {
          setConsentFormIsOpen(false);
        },
      });
    }
  }, [isOnTosPage]);

  React.useEffect(() => {
    if (settings?.is_new_user && config.data?.APP_MODE === "saas") {
      displaySuccessToast(t(I18nKey.BILLING$YOURE_IN));
    }
  }, [settings?.is_new_user, config.data?.APP_MODE]);

  React.useEffect(() => {
    // Don't do any redirects when on TOS page
    // Don't allow users to use the app if it 402s
    if (!isOnTosPage && error?.status === 402 && pathname !== "/") {
      navigate("/");
    }
  }, [error?.status, pathname, isOnTosPage]);

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

  const renderAuthModal =
    !isAuthed &&
    !isAuthError &&
    !isFetchingAuth &&
    !isOnTosPage &&
    config.data?.APP_MODE === "saas" &&
    !loginMethodExists; // Don't show auth modal if login method exists in local storage

  const renderReAuthModal =
    !isAuthed &&
    !isAuthError &&
    !isFetchingAuth &&
    !isOnTosPage &&
    config.data?.APP_MODE === "saas" &&
    loginMethodExists;

  return (
    <div
      data-testid="root-layout"
      className={cn(
        "h-screen lg:min-w-[1024px] flex flex-col md:flex-row bg-base",
        pathname === "/" ? "p-0" : "p-0 md:p-3 md:pl-0",
        isMobileDevice() && "overflow-hidden",
      )}
    >
      <Sidebar />

      <div className="flex flex-col w-full h-[calc(100%-50px)] md:h-full gap-3">
        {config.data?.MAINTENANCE && (
          <MaintenanceBanner startTime={config.data.MAINTENANCE.startTime} />
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

      {renderAuthModal && (
        <AuthModal
          githubAuthUrl={effectiveGitHubAuthUrl}
          appMode={config.data?.APP_MODE}
          providersConfigured={config.data?.PROVIDERS_CONFIGURED}
          authUrl={config.data?.AUTH_URL}
        />
      )}
      {renderReAuthModal && <ReauthModal />}
      {config.data?.APP_MODE === "oss" && consentFormIsOpen && (
        <AnalyticsConsentFormModal
          onClose={() => {
            setConsentFormIsOpen(false);
          }}
        />
      )}

      {config.data?.FEATURE_FLAGS.ENABLE_BILLING &&
        config.data?.APP_MODE === "saas" &&
        settings?.is_new_user && <SetupPaymentModal />}
    </div>
  );
}
