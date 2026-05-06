import { QueryCache, MutationCache, QueryClient } from "@tanstack/react-query";
import i18next from "i18next";
import { AxiosError } from "axios";
import { I18nKey } from "./i18n/declaration";
import { retrieveAxiosErrorMessage } from "./utils/retrieve-axios-error-message";
import { displayErrorToast } from "./utils/custom-toast-handlers";

const handle401Error = (error: AxiosError, queryClient: QueryClient) => {
  if (error?.response?.status === 401 || error?.status === 401) {
    queryClient.invalidateQueries({ queryKey: ["user", "authenticated"] });
  }
};

// While a single-use handoff code is still being redeemed, the SPA may fire
// API calls before the cookie lands and get a 401 back. The redemption hook
// refetches those errored queries afterward, so the toast is just noise.
const isHandoffInFlight = (): boolean => {
  if (typeof window === "undefined") return false;
  return new URLSearchParams(window.location.search).has("handoff_code");
};

const shownErrors = new Set<string>();
export const queryClient = new QueryClient({
  queryCache: new QueryCache({
    onError: (error, query) => {
      const isAuthQuery =
        query.queryKey[0] === "user" && query.queryKey[1] === "authenticated";
      if (!isAuthQuery) {
        handle401Error(error, queryClient);
      }

      const is401 =
        (error as AxiosError)?.response?.status === 401 ||
        (error as AxiosError)?.status === 401;

      if (!query.meta?.disableToast && !(is401 && isHandoffInFlight())) {
        const errorMessage = retrieveAxiosErrorMessage(error);

        if (!shownErrors.has(errorMessage || "")) {
          displayErrorToast(errorMessage || i18next.t(I18nKey.ERROR$GENERIC));
          shownErrors.add(errorMessage);

          setTimeout(() => {
            shownErrors.delete(errorMessage);
          }, 3000);
        }
      }
    },
  }),
  mutationCache: new MutationCache({
    onError: (error, _, __, mutation) => {
      handle401Error(error, queryClient);

      if (!mutation?.meta?.disableToast) {
        const message = retrieveAxiosErrorMessage(error);
        displayErrorToast(message || i18next.t(I18nKey.ERROR$GENERIC));
      }
    },
  }),
});
