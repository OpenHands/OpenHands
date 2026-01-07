import { useEffect, useState, useCallback } from "react";

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
        // eslint-disable-next-line no-console
        console.error("reCAPTCHA execution failed:", err);
        return null;
      }
    },
    [siteKey, isReady],
  );

  return { isReady, isLoading, error, executeRecaptcha };
}
