import { useCallback, useEffect, useMemo, useRef } from "react";
import axios from "axios";
import AutomationService from "#/api/automation-service/automation-service.api";
import { isSdkHttpStatusError } from "#/api/agent-server-compatibility";
import { useActiveBackend } from "#/contexts/active-backend-context";
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
 * Resolves to null only when the entry has no service draft to check. A 404 or
 * 501 is the explicit legacy-deployment advisory path; malformed responses,
 * transport errors, and service failures return an unavailable outcome that
 * blocks creation. Superseded requests return stale so an older response can
 * never overwrite the latest verdict.
 */
export type SetupPreflightOutcome =
  | { status: "passed" }
  | { status: "failed"; errors: MappedManifestErrors }
  | { status: "unsupported" }
  | { status: "unavailable" }
  | { status: "stale" };

function hasMappedErrors(errors: MappedManifestErrors): boolean {
  return (
    errors.formErrors.length > 0 ||
    Object.keys(errors.fieldErrors).length > 0 ||
    Object.values(errors.stepErrors).some((messages) => messages?.length)
  );
}

export function useSetupPreflight(entry: SetupEntry) {
  const { backend, orgId } = useActiveBackend();
  const latestRequestRef = useRef(0);
  useEffect(
    () => () => {
      latestRequestRef.current += 1;
    },
    [],
  );
  const currentEntryIdRef = useRef(entry.id);
  currentEntryIdRef.current = entry.id;
  const targetKey = JSON.stringify([
    backend.id,
    backend.kind,
    backend.connectionRevision ?? 0,
    orgId,
  ]);
  const currentTargetKeyRef = useRef(targetKey);
  currentTargetKeyRef.current = targetKey;
  const errorMap = useMemo(() => deriveErrorMap(entry), [entry]);

  const runPreflight = useCallback(
    async (
      formValues: SetupFormValues,
    ): Promise<SetupPreflightOutcome | null> => {
      const body = buildPreflightBody(entry, formValues);
      if (!body) return null;

      latestRequestRef.current += 1;
      const requestId = latestRequestRef.current;
      const entryId = entry.id;
      const requestTargetKey = targetKey;

      const isStale = () =>
        requestId !== latestRequestRef.current ||
        entryId !== currentEntryIdRef.current ||
        requestTargetKey !== currentTargetKeyRef.current;

      try {
        const result = await AutomationService.validateDraft(body);
        if (isStale()) return { status: "stale" };
        if (
          result?.valid === true &&
          Array.isArray(result.errors) &&
          result.errors.length === 0
        ) {
          return { status: "passed" };
        }

        const errors = mapServiceErrors(
          normalizeServiceErrors(result, body.draft as SetupRequestBody),
          errorMap,
        );
        if (result?.valid === false && hasMappedErrors(errors)) {
          return { status: "failed", errors };
        }
        return { status: "unavailable" };
      } catch (error) {
        if (isStale()) return { status: "stale" };
        return isPreflightUnimplemented(error)
          ? { status: "unsupported" }
          : { status: "unavailable" };
      }
    },
    [entry, errorMap, targetKey],
  );

  const invalidatePreflight = useCallback(() => {
    latestRequestRef.current += 1;
  }, []);

  return { runPreflight, invalidatePreflight };
}
