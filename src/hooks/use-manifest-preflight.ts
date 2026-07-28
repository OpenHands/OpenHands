import { useCallback, useRef } from "react";
import AutomationService from "#/api/automation-service/automation-service.api";
import {
  mapServiceErrors,
  normalizeServiceErrors,
  type MappedManifestErrors,
} from "#/manifests/manifest-error-map";
import { buildRequestBody } from "#/manifests/manifest-template";
import type {
  ExtensionManifest,
  ManifestFormValues,
  ManifestRequestBody,
} from "#/manifests/types";

const NO_ERRORS: MappedManifestErrors = { fieldErrors: {}, formErrors: [] };

/**
 * Stage 6 — ask the service whether a draft is valid before anything is created.
 *
 * The service is the authoritative validator, so what it receives is the mapped
 * payload rather than the raw form values: what is checked is exactly what
 * would be sent. Errors come back addressed by payload path and are translated
 * back to fields through the manifest's `errorMap`.
 *
 * Resolves to null when there is no verdict — the manifest declares no
 * preflight, the deployment does not implement one, or a newer run has already
 * superseded this one. A missing preflight is not a failure: local checks and
 * the create response still stand between the user and a bad configuration.
 */
export function useManifestPreflight(manifest: ExtensionManifest) {
  const latestRequestRef = useRef(0);

  return useCallback(
    async (
      formValues: ManifestFormValues,
      payload: ManifestRequestBody | null,
    ): Promise<MappedManifestErrors | null> => {
      const validation = manifest.validation;
      const preflight = validation?.preflight;
      if (!validation || !preflight) return null;

      const body = buildRequestBody(preflight.body, {
        manifest,
        form: formValues,
        submit: { payload: payload ?? undefined },
      });

      latestRequestRef.current += 1;
      const requestId = latestRequestRef.current;

      try {
        const result = await AutomationService.validateDraft(
          preflight.path,
          body,
        );
        if (requestId !== latestRequestRef.current) return null;
        if (result?.valid) return NO_ERRORS;

        return mapServiceErrors(
          normalizeServiceErrors(result, payload),
          validation.onInvalid.errorMap,
          validation.onInvalid.errorTarget,
        );
      } catch {
        return null;
      }
    },
    [manifest],
  );
}
