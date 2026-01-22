import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import GoogleLogo from "#/assets/branding/google-logo.svg?react";

interface GoogleCredentialResponse {
  clientId?: string; // Google OAuth client ID
  credential: string; // JWT ID token
  select_by?: string; // how the user selected this account
  momentType?: string;
}

interface GoogleIdentity {
  accounts: {
    id: {
      initialize: (options: {
        client_id: string;
        callback: (response: GoogleCredentialResponse) => void;
        auto_select?: boolean;
        cancel_on_tap_outside?: boolean;
        prompt_parent_id?: string;
      }) => void;

      renderButton: (
        parent: HTMLElement,
        options: {
          type?: "standard" | "icon";
          theme?: "outline" | "filled_blue" | "filled_black";
          size?: "small" | "medium" | "large";
          text?: "signin_with" | "signup_with" | "continue_with" | "signin";
          shape?: "rectangular" | "pill" | "circle";
          logo_alignment?: "left" | "center";
          width?: number;
          locale?: string;
        },
      ) => void;

      prompt: () => void;
      cancel: () => void;
    };
  };
}

declare global {
  interface Window {
    google?: GoogleIdentity;
  }
}

interface GoogleSignInButtonProps {
  clientId: string;
  onCredentialResponse: (response: GoogleCredentialResponse) => void;
}

export default function GoogleSignInButton({
  clientId,
  onCredentialResponse,
}: GoogleSignInButtonProps) {
  const { t } = useTranslation();

  // Load GSI script once
  useEffect(() => {
    if (window.google) return;

    const script = document.createElement("script");
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.defer = true;
    document.body.appendChild(script);
  }, []);

  // Handle button click
  const handleClick = () => {
    if (!window.google) {
      return;
    }

    window.google.accounts.id.initialize({
      client_id: clientId,
      callback: onCredentialResponse,
    });

    // Show the One Tap / prompt
    window.google.accounts.id.prompt();
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      className="w-full flex items-center justify-center gap-1.5 rounded border border-gray-300 h-10 bg-white hover:bg-gray-100"
    >
      <GoogleLogo className="w-5 h-5" />
      <span className="text-sm font-medium text-gray-700">
        {t(I18nKey.AUTH$LOGIN_WITH_GOOGLE)}
      </span>
    </button>
  );
}
