import { useCallback, useMemo, useRef } from "react";
import AutomationService from "#/api/automation-service/automation-service.api";
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

/**
 * Stage 6 — ask the service whether a draft is valid before anything is created.
 *
 * The service is the authoritative validator, so what it receives is the derived
 * payload rather than the raw form values: what is checked is exactly what would
 * be sent. Errors come back addressed by payload path and are translated back to
 * fields through the map derived from that same builder.
 *
 * Resolves to null when there is no verdict — the entry has no draft to check,
 * the deployment does not implement preflight, or a newer run has already
 * superseded this one. A missing preflight is not a failure: local checks and
 * the create response still stand between the user and a bad configuration.
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
      } catch {
        return null;
      }
    },
    [entry, errorMap],
  );
}
