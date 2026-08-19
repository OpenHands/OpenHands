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
  ValidateDraftResponse,
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
 * 501 is the explicit legacy-deployment advisory path. An older endpoint that
 * explicitly rejects the additive `requirements` field gets one legacy-body
 * retry and is also advisory only when that retry validates the draft.
 * Malformed responses, transport errors, and service failures return an
 * unavailable outcome that blocks creation. Superseded requests return stale
 * so an older response can never overwrite the latest verdict.
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/**
 * Older validate endpoints reject the additive `requirements` envelope field
 * with Pydantic's explicit extra-field error. Only that exact shape is safe to
 * retry without the field; a different 422 remains a real service failure.
 */
function isRequirementsSchemaError(error: unknown): boolean {
  const status = axios.isAxiosError(error)
    ? error.response?.status
    : isSdkHttpStatusError(error, 422)
      ? 422
      : undefined;
  if (status !== 422) return false;

  const body = axios.isAxiosError(error)
    ? error.response?.data
    : isRecord(error)
      ? error.response
      : undefined;
  const detail = isRecord(body) && "detail" in body ? body.detail : body;
  if (!Array.isArray(detail)) return false;

  return detail.some((item) => {
    if (!isRecord(item)) return false;
    const loc = item.loc;
    const type = item.type;
    return (
      Array.isArray(loc) &&
      loc.length === 2 &&
      loc[0] === "body" &&
      loc[1] === "requirements" &&
      (type === "extra_forbidden" || type === "value_error.extra")
    );
  });
}

function withoutRequirements(body: SetupRequestBody): SetupRequestBody {
  return Object.fromEntries(
    Object.entries(body).filter(([key]) => key !== "requirements"),
  );
}

async function validateDraftWithCompatibility(
  body: SetupRequestBody,
  isStale: () => boolean,
): Promise<{ result: ValidateDraftResponse; usedLegacyContract: boolean }> {
  try {
    return {
      result: await AutomationService.validateDraft(body),
      usedLegacyContract: false,
    };
  } catch (error) {
    if (!isRequirementsSchemaError(error) || isStale()) throw error;

    return {
      result: await AutomationService.validateDraft(withoutRequirements(body)),
      usedLegacyContract: true,
    };
  }
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
  // Catalog entries are JSON-shaped data. A signature keeps equivalent
  // rematerialized objects from invalidating a request, while still detecting
  // a same-id entry whose setup contract was refreshed.
  const entryKey = JSON.stringify(entry);
  const currentEntryKeyRef = useRef(entryKey);
  currentEntryKeyRef.current = entryKey;
  const targetKey = JSON.stringify([
    backend.id,
    backend.kind,
    backend.host,
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
      // Every invocation supersedes an earlier one, including an assisted
      // entry that has no draft to send. This closes the old request before
      // checking whether the current entry needs a service call.
      latestRequestRef.current += 1;
      const requestId = latestRequestRef.current;
      const requestEntryKey = entryKey;
      const requestTargetKey = targetKey;
      const body = buildPreflightBody(entry, formValues);
      if (!body) return null;

      const isStale = () =>
        requestId !== latestRequestRef.current ||
        requestEntryKey !== currentEntryKeyRef.current ||
        requestTargetKey !== currentTargetKeyRef.current;

      try {
        const { result, usedLegacyContract } =
          await validateDraftWithCompatibility(body, isStale);
        if (isStale()) return { status: "stale" };
        if (
          result?.valid === true &&
          Array.isArray(result.errors) &&
          result.errors.length === 0
        ) {
          return {
            status: usedLegacyContract ? "unsupported" : "passed",
          };
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
