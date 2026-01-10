import { useEffect, useState, useCallback } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

const RECAPTCHA_SCRIPT_URL = "https://www.google.com/recaptcha/enterprise.js";

interface UseRecaptchaOptions {
  siteKey?: string;
}

export interface UseRecaptchaReturn {
  isReady: boolean;
  isLoading: boolean;
  error: Error | null;
  executeRecaptcha: (action: string) => Promise<string | null>;
}

export function useRecaptcha({
  siteKey,
}: UseRecaptchaOptions): UseRecaptchaReturn {
  const { t } = useTranslation();
  const [isReady, setIsReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!siteKey) return;

    // Check if script is already loaded
    if (window.grecaptcha?.enterprise) {
      window.grecaptcha.enterprise.ready(() => setIsReady(true));
      return;
    }

    setIsLoading(true);

    const script = document.createElement("script");
    script.src = `${RECAPTCHA_SCRIPT_URL}?render=${siteKey}`;
    script.async = true;
    script.defer = true;

    script.onload = () => {
      window.grecaptcha?.enterprise.ready(() => {
        setIsReady(true);
        setIsLoading(false);
      });
    };

    script.onerror = () => {
      setError(new Error("Failed to load reCAPTCHA script"));
      setIsLoading(false);
    };

    document.head.appendChild(script);
  }, [siteKey]);

  const executeRecaptcha = useCallback(
    async (action: string): Promise<string | null> => {
      if (!siteKey || !isReady || !window.grecaptcha?.enterprise) return null;

      try {
        const token = await window.grecaptcha.enterprise.execute(siteKey, {
          action,
        });
        return token;
      } catch (err) {
        displayErrorToast(t(I18nKey.AUTH$RECAPTCHA_BLOCKED));
        return null;
      }
    },
    [siteKey, isReady, t],
  );

  return { isReady, isLoading, error, executeRecaptcha };
}
