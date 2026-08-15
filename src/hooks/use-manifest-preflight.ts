import { useCallback, useMemo, useRef } from "react";
import axios from "axios";
import AutomationService from "#/api/automation-service/automation-service.api";
import { isSdkHttpStatusError } from "#/api/agent-server-compatibility";
import {
  mapServiceErrors,
  normalizeServiceErrors,
  type MappedManifestErrors,
} from "#/manifests/manifest-error-map";
import {
  buildPreflightBody,
  deriveErrorMap,
} from "#/manifests/automation-setup";
import type {
  SetupEntry,
  SetupFormValues,
  SetupRequestBody,
} from "#/manifests/types";

const NO_ERRORS: MappedManifestErrors = { fieldErrors: {}, formErrors: [] };

/** What a deployment that does not serve the validate endpoint answers with. */
const NOT_IMPLEMENTED_STATUSES = [404, 501];

/**
 * Whether a failed preflight means "this deployment does not implement it".
 * Local calls throw an `AxiosError` and cloud calls throw the shared client's
 * `HttpError`, so both shapes are read.
 */
function isPreflightUnimplemented(error: unknown): boolean {
  return NOT_IMPLEMENTED_STATUSES.some(
    (status) =>
      isSdkHttpStatusError(error, status) ||
      (axios.isAxiosError(error) && error.response?.status === status),
  );
}

/**
 * Stage 6 — ask the service whether a draft is valid before anything is created.
 *
 * The service is the authoritative validator, so what it receives is the derived
 * payload rather than the raw form values: what is checked is exactly what would
 * be sent. Errors come back addressed by payload path and are translated back to
 * fields through the map derived from that same builder.
 *
 * Resolves to null when there is no verdict — the entry has no draft to check,
 * the deployment does not implement preflight, a newer run has already
 * superseded this one, or the request failed. A missing preflight is not a
 * failure: local checks and the create response still stand between the user
 * and a bad configuration. Only a deployment without the endpoint is an
 * expected failure though, so any other one is reported.
 */
export function useSetupPreflight(entry: SetupEntry) {
  const latestRequestRef = useRef(0);
  const errorMap = useMemo(() => deriveErrorMap(entry), [entry]);

  return useCallback(
    async (
      formValues: SetupFormValues,
    ): Promise<MappedManifestErrors | null> => {
      const body = buildPreflightBody(entry, formValues);
      if (!body) return null;

      latestRequestRef.current += 1;
      const requestId = latestRequestRef.current;

      try {
        const result = await AutomationService.validateDraft(body);
        if (requestId !== latestRequestRef.current) return null;
        if (result?.valid) return NO_ERRORS;

        return mapServiceErrors(
          normalizeServiceErrors(result, body.draft as SetupRequestBody),
          errorMap,
        );
      } catch (error) {
        // A broken validator and an unimplemented one degrade to the same
        // advisory "no verdict", so the only thing that separates them is this
        // line. Without it a service returning 500 on every draft is invisible.
        if (!isPreflightUnimplemented(error)) {
          console.warn("Automation setup preflight failed:", error);
        }
        return null;
      }
    },
    [entry, errorMap],
  );
}
