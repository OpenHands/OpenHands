import "@tanstack/react-query";
import type { AxiosError } from "axios";

interface MyMeta extends Record<string, unknown> {
  disableToast?: boolean;
  /**
   * When set on a query that targets a specific local backend, a successful
   * fetch automatically clears that backend's failure counter in the health
   * store — re-arming the status dot without requiring the user to edit the
   * backend's config.
   */
  backendId?: string;
}

declare module "@tanstack/react-query" {
  interface Register {
    defaultError: AxiosError;

    queryMeta: MyMeta;
    mutationMeta: MyMeta;
  }
}
