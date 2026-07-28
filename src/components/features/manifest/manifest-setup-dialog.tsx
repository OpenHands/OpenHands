import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { modalTitleLgClassName } from "#/utils/modal-classes";
import { getApiErrorBody } from "#/utils/api-error-message";
import { useManifestCapabilities } from "#/hooks/query/use-manifest-capabilities";
import { useManifestPrerequisites } from "#/hooks/query/use-manifest-prerequisites";
import { useManifestPreflight } from "#/hooks/use-manifest-preflight";
import { useManifestAnalytics } from "#/hooks/use-manifest-analytics";
import {
  buildManifestPayload,
  useManifestAction,
} from "#/manifests/manifest-actions";
import {
  getFieldOptions,
  getInitialFormValues,
  resolveFieldOverrides,
  validateFormValues,
  type ManifestFieldError,
  type ManifestFieldErrors,
} from "#/manifests/manifest-local-validation";
import {
  mapServiceErrors,
  normalizeServiceErrors,
  type MappedManifestErrors,
} from "#/manifests/manifest-error-map";
import {
  interpolateText,
  type ManifestScope,
} from "#/manifests/manifest-template";
import type { GitRepository } from "#/types/git";
import type { ExtensionManifest, ManifestFormValues } from "#/manifests/types";
import { ManifestFormField } from "./manifest-form-field";
import { ManifestPrerequisitesStep } from "./manifest-prerequisites-step";
import { ManifestReviewStep } from "./manifest-review-step";

type SetupStep = "prerequisites" | "form" | "review";

const NO_SERVICE_ERRORS: MappedManifestErrors = {
  fieldErrors: {},
  formErrors: [],
};

function hasAnyError(errors: MappedManifestErrors): boolean {
  return (
    errors.formErrors.length > 0 || Object.keys(errors.fieldErrors).length > 0
  );
}

export interface ManifestSetupDialogProps {
  manifest: ExtensionManifest;
  onClose: () => void;
}

/**
 * The generic setup host: capabilities check, prerequisites, form, review, and
 * the manifest-declared action, rendered as a dialog.
 *
 * The host runs the stages; the manifest supplies every stage's content. Any
 * string the user reads here is either manifest-authored or host chrome, and
 * the host never adds a word about what is being configured.
 */
export function ManifestSetupDialog({
  manifest,
  onClose,
}: ManifestSetupDialogProps) {
  const { t } = useTranslation("openhands");
  const navigate = useNavigate();

  const capabilities = useManifestCapabilities(manifest);
  const prerequisites = useManifestPrerequisites(manifest);
  const runPreflight = useManifestPreflight(manifest);
  const runAction = useManifestAction();
  const emitStage = useManifestAnalytics(manifest);

  const [step, setStep] = useState<SetupStep>("prerequisites");
  const [values, setValues] = useState<ManifestFormValues>(() =>
    getInitialFormValues(manifest),
  );
  const [repositories, setRepositories] = useState<
    Record<string, GitRepository | null>
  >({});
  const [localErrors, setLocalErrors] = useState<ManifestFieldErrors>({});
  const [serviceErrors, setServiceErrors] =
    useState<MappedManifestErrors>(NO_SERVICE_ERRORS);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Blur-triggered preflight reads the values as they are when the field is
  // left, which can be the same tick as the change that caused it.
  const valuesRef = useRef(values);
  const blurTimerRef = useRef<number | null>(null);
  useEffect(
    () => () => {
      if (blurTimerRef.current) window.clearTimeout(blurTimerRef.current);
    },
    [],
  );

  const capabilitiesScope = useMemo(
    () => ({
      ...(capabilities.capabilities ?? {}),
      supported: capabilities.supported,
    }),
    [capabilities.capabilities, capabilities.supported],
  );

  const overrides = useMemo(
    () => resolveFieldOverrides(manifest, capabilities.capabilities),
    [manifest, capabilities.capabilities],
  );

  const payload = useMemo(
    () => buildManifestPayload(manifest, values),
    [manifest, values],
  );

  const scope: ManifestScope = useMemo(
    () => ({
      manifest,
      form: values,
      capabilities: capabilitiesScope,
      submit: { payload: payload ?? undefined },
    }),
    [manifest, values, capabilitiesScope, payload],
  );

  const emittedRouteRef = useRef(false);
  useEffect(() => {
    if (emittedRouteRef.current) return;
    emittedRouteRef.current = true;
    emitStage("route.entered");
  }, [emitStage]);

  const emittedCapabilitiesRef = useRef(false);
  useEffect(() => {
    if (capabilities.isLoading || emittedCapabilitiesRef.current) return;
    emittedCapabilitiesRef.current = true;
    emitStage("capabilities.resolved", { capabilities: capabilitiesScope });
  }, [capabilities.isLoading, capabilitiesScope, emitStage]);

  const isUnsupported = capabilities.supported === false;
  const showPrerequisites =
    prerequisites.blockingIntegrations.length > 0 ||
    prerequisites.warningIntegrations.length > 0 ||
    prerequisites.missingSecrets.length > 0;
  // Prerequisites resolve asynchronously; deriving the step keeps a manifest
  // with nothing to check from flashing an empty first screen.
  const currentStep: SetupStep =
    step === "prerequisites" && !showPrerequisites ? "form" : step;

  const setFieldValue = (name: string, value: string) => {
    valuesRef.current = { ...valuesRef.current, [name]: value };
    setValues(valuesRef.current);
    setLocalErrors(({ [name]: _removed, ...rest }) => rest);
    setServiceErrors((current) => {
      if (!(name in current.fieldErrors)) return current;
      const { [name]: _cleared, ...fieldErrors } = current.fieldErrors;
      return { ...current, fieldErrors };
    });
  };

  const handleFieldBlur = () => {
    const preflight = manifest.validation?.preflight;
    if (!preflight?.runOn.includes("fieldBlur")) return;

    if (blurTimerRef.current) window.clearTimeout(blurTimerRef.current);
    blurTimerRef.current = window.setTimeout(() => {
      const draft = valuesRef.current;
      void runPreflight(draft, buildManifestPayload(manifest, draft)).then(
        (result) => {
          if (result) setServiceErrors(result);
        },
      );
    }, preflight.debounceMs ?? 0);
  };

  const handleContinue = async () => {
    if (currentStep === "prerequisites") {
      setStep("form");
      return;
    }

    const failures = validateFormValues(manifest, values, overrides);
    if (Object.keys(failures).length > 0) {
      setLocalErrors(failures);
      return;
    }
    setLocalErrors({});

    if (manifest.validation?.preflight?.runOn.includes("beforeSubmit")) {
      const result = await runPreflight(values, payload);
      if (result && hasAnyError(result)) {
        setServiceErrors(result);
        return;
      }
    }

    setServiceErrors(NO_SERVICE_ERRORS);
    emitStage("validation.succeeded", { form: values });
    setStep("review");
  };

  const handleConfirm = async () => {
    setIsSubmitting(true);
    try {
      const { response } = await runAction(manifest, values, payload);
      emitStage("submit.succeeded", { form: values, response });
      navigate(
        interpolateText(manifest.submit.onSuccess.to, { ...scope, response }),
        { replace: true },
      );
    } catch (error) {
      emitStage("submit.failed", { form: values });
      const { onError } = manifest.submit;
      const mapped = mapServiceErrors(
        normalizeServiceErrors(getApiErrorBody(error), payload),
        onError.reuseErrorMap
          ? manifest.validation?.onInvalid.errorMap
          : undefined,
        onError.errorTarget,
      );
      setServiceErrors(
        hasAnyError(mapped)
          ? mapped
          : {
              fieldErrors: {},
              formErrors: [onError.message ?? t(I18nKey.SETUP$SUBMIT_FAILED)],
            },
      );
      setStep("form");
    } finally {
      setIsSubmitting(false);
    }
  };

  const resolveFieldError = (name: string): string | undefined => {
    const local = localErrors[name];
    return local ? formatFieldError(local, t) : serviceErrors.fieldErrors[name];
  };

  const isLoading = capabilities.isLoading || prerequisites.isLoading;

  const title = (() => {
    if (isUnsupported) return t(I18nKey.SETUP$UNAVAILABLE_TITLE);
    if (currentStep === "prerequisites")
      return t(I18nKey.SETUP$PREREQUISITES_TITLE);
    if (currentStep === "review") return manifest.review.title;
    return manifest.name;
  })();

  return (
    <ModalBackdrop onClose={onClose} aria-label={manifest.name}>
      <div
        data-testid="manifest-setup-dialog"
        className="relative flex max-h-[85vh] w-[92vw] max-w-lg flex-col rounded-xl border border-[var(--oh-border)] bg-base-secondary"
      >
        <ModalCloseButton
          onClose={onClose}
          testId="manifest-setup-dialog-close"
          disabled={isSubmitting}
        />
        <header className="flex-shrink-0 px-6 pb-4 pt-6">
          <h2 className={cn("pr-6", modalTitleLgClassName)}>{title}</h2>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto px-6">
          {isLoading && (
            <div className="flex justify-center py-6">
              <LoadingSpinner size="small" />
            </div>
          )}

          {!isLoading && isUnsupported && (
            <p className="text-sm text-[var(--oh-muted)]">
              {manifest.capabilities?.onUnsupported.message}
            </p>
          )}

          {!isLoading && !isUnsupported && currentStep === "prerequisites" && (
            <ManifestPrerequisitesStep
              requires={manifest.requires!}
              prerequisites={prerequisites}
            />
          )}

          {!isLoading && !isUnsupported && currentStep === "form" && (
            <div className="flex flex-col gap-5">
              <p className="text-sm text-[var(--oh-muted)]">
                {manifest.description}
              </p>
              {manifest.form.note && (
                <p className="text-sm text-[var(--oh-muted)]">
                  {manifest.form.note}
                </p>
              )}
              {manifest.form.fields.map((field) => (
                <ManifestFormField
                  key={field.name}
                  field={field}
                  value={values[field.name] ?? ""}
                  error={resolveFieldError(field.name)}
                  options={getFieldOptions(field, overrides)}
                  repository={repositories[field.name] ?? null}
                  disabled={isSubmitting}
                  onChange={(value) => setFieldValue(field.name, value)}
                  onRepositoryChange={(repository) =>
                    setRepositories((current) => ({
                      ...current,
                      [field.name]: repository,
                    }))
                  }
                  onBlur={handleFieldBlur}
                />
              ))}
            </div>
          )}

          {!isLoading && !isUnsupported && currentStep === "review" && (
            <ManifestReviewStep review={manifest.review} scope={scope} />
          )}

          {serviceErrors.formErrors.map((message) => (
            <p
              key={message}
              role="alert"
              data-testid="manifest-form-error"
              className="pt-4 text-sm text-red-400"
            >
              {message}
            </p>
          ))}
        </div>

        <footer className="flex flex-shrink-0 justify-end gap-2 px-6 pb-6 pt-4">
          {currentStep === "review" && (
            <BrandButton
              testId="manifest-back-button"
              type="button"
              variant="secondary"
              isDisabled={isSubmitting}
              onClick={() => setStep("form")}
            >
              {t(I18nKey.BUTTON$BACK)}
            </BrandButton>
          )}

          {isUnsupported ? (
            <BrandButton type="button" variant="secondary" onClick={onClose}>
              {t(I18nKey.BUTTON$CLOSE)}
            </BrandButton>
          ) : (
            <BrandButton
              testId="manifest-continue-button"
              type="button"
              variant="primary"
              isDisabled={
                isLoading ||
                isSubmitting ||
                (currentStep === "prerequisites" && prerequisites.isBlocked)
              }
              onClick={
                currentStep === "review" ? handleConfirm : handleContinue
              }
            >
              {currentStep === "review"
                ? manifest.review.confirmLabel
                : t(I18nKey.BUTTON$CONTINUE)}
            </BrandButton>
          )}
        </footer>
      </div>
    </ModalBackdrop>
  );
}

function formatFieldError(
  error: ManifestFieldError,
  t: (key: I18nKey, options?: Record<string, unknown>) => string,
): string {
  switch (error.code) {
    case "minLength":
      return t(I18nKey.SETUP$VALIDATION_MIN_LENGTH, { length: error.length });
    case "maxLength":
      return t(I18nKey.SETUP$VALIDATION_MAX_LENGTH, { length: error.length });
    case "invalidOption":
      return t(I18nKey.SETUP$VALIDATION_INVALID_OPTION);
    case "unsafeExpressionLiteral":
      return t(I18nKey.SETUP$VALIDATION_UNSAFE_VALUE);
    default:
      return t(I18nKey.SETUP$VALIDATION_REQUIRED);
  }
}
