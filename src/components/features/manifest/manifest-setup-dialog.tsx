import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router";
import AutomationService from "#/api/automation-service/automation-service.api";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { modalTitleLgClassName } from "#/utils/modal-classes";
import {
  getApiErrorBody,
  getApiErrorMessage,
} from "#/utils/api-error-message";
import { useTracking } from "#/hooks/use-tracking";
import { useAutomationRuns } from "#/hooks/query/use-automation-detail";
import { useSetupCapabilities } from "#/hooks/query/use-manifest-capabilities";
import { useSetupPrerequisites } from "#/hooks/query/use-manifest-prerequisites";
import { useSetupPreflight } from "#/hooks/use-manifest-preflight";
import { useSetupAction } from "#/manifests/manifest-actions";
import {
  buildCreatePayload,
  deriveErrorMap,
  missingCreateEndpoints,
} from "#/manifests/automation-setup";
import {
  collectFields,
  getFieldOptions,
  getInitialFormValues,
  resolveFieldOverrides,
  validateFormValues,
  type SetupFieldError,
  type SetupFieldErrors,
} from "#/manifests/manifest-local-validation";
import {
  mapServiceErrors,
  normalizeServiceErrors,
  type MappedManifestErrors,
} from "#/manifests/manifest-error-map";
import { automationDetailPath } from "#/manifests/automation-interface";
import { findAutomationCommand } from "#/utils/automation-catalog";
import {
  AutomationRunStatus,
  type AutomationRun,
} from "#/types/automation";
import type { GitRepository } from "#/types/git";
import type {
  SetupEntry,
  SetupFormValue,
  SetupFormValues,
  SetupMode,
  SetupRequestBody,
} from "#/manifests/types";
import { SetupFormField } from "./manifest-form-field";
import { SetupPrerequisitesStep } from "./manifest-prerequisites-step";
import { SetupReviewStep } from "./manifest-review-step";
import { SetupTestRunStep } from "./manifest-setup-test-run-step";

type SetupStep = "prerequisites" | "form" | "review" | "test";

/** How long a field rests after a blur before the draft is sent for checking. */
const PREFLIGHT_DEBOUNCE_MS = 400;

const NO_SERVICE_ERRORS: MappedManifestErrors = {
  fieldErrors: {},
  formErrors: [],
};

function hasAnyError(errors: MappedManifestErrors): boolean {
  return (
    errors.formErrors.length > 0 || Object.keys(errors.fieldErrors).length > 0
  );
}

/** Where a completed setup lands: the new automation, or the conversation. */
function getDestination(response: Record<string, unknown>): string | null {
  if (typeof response.conversation_id === "string") {
    return `/conversations/${response.conversation_id}`;
  }
  if (typeof response.id === "string") return automationDetailPath(response.id);
  return null;
}

function isRunInFlight(run: AutomationRun | null): boolean {
  return (
    run?.status === AutomationRunStatus.PENDING ||
    run?.status === AutomationRunStatus.RUNNING
  );
}

export interface SetupDialogProps {
  entry: SetupEntry;
  onClose: () => void;
}

/**
 * The setup host: capabilities check, prerequisites, form, review, a controlled
 * test run for newly-created direct automations, and the action the manifest's
 * mode selects, rendered as a dialog.
 *
 * The host runs the stages and owns everything the same for every entry; the
 * manifest supplies only what varies. Any string the user reads here is either
 * manifest-authored or host chrome.
 */
export function SetupDialog({ entry, onClose }: SetupDialogProps) {
  const { t } = useTranslation("openhands");
  const navigate = useNavigate();

  const capabilities = useSetupCapabilities(entry);
  const prerequisites = useSetupPrerequisites(entry);
  const runPreflight = useSetupPreflight(entry);
  const runAction = useSetupAction();
  const {
    trackAutomationSetupOpened,
    trackAutomationSetupValidated,
    trackAutomationSetupCreated,
    trackAutomationSetupFailed,
  } = useTracking();

  const [step, setStep] = useState<SetupStep>("prerequisites");
  const [values, setValues] = useState<SetupFormValues>(() =>
    getInitialFormValues(entry.setup),
  );
  const [repositories, setRepositories] = useState<
    Record<string, GitRepository | null>
  >({});
  const [localErrors, setLocalErrors] = useState<SetupFieldErrors>({});
  const [serviceErrors, setServiceErrors] =
    useState<MappedManifestErrors>(NO_SERVICE_ERRORS);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [createdAutomationId, setCreatedAutomationId] = useState<string | null>(
    null,
  );
  const [testRunId, setTestRunId] = useState<string | null>(null);
  const [dispatchedRun, setDispatchedRun] = useState<AutomationRun | null>(null);
  const [isDispatchingTest, setIsDispatchingTest] = useState(false);
  const [isFinalizing, setIsFinalizing] = useState(false);

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

  const fields = useMemo(() => collectFields(entry.setup), [entry]);
  const overrides = useMemo(
    () => resolveFieldOverrides(entry.setup, capabilities.capabilities),
    [entry, capabilities.capabilities],
  );
  const payload = useMemo(
    () => buildCreatePayload(entry, values),
    [entry, values],
  );
  const errorMap = useMemo(() => deriveErrorMap(entry), [entry]);

  const emittedOpenRef = useRef(false);
  useEffect(() => {
    if (emittedOpenRef.current) return;
    emittedOpenRef.current = true;
    trackAutomationSetupOpened({ automationId: entry.id });
  }, [entry.id, trackAutomationSetupOpened]);

  // An entry the published interface cannot create is refused here rather than
  // at the moment of creating: a bundle needs two endpoints a manifest from
  // before bundles does not declare, and no answer the user gives supplies
  // them. Named alongside the deployment's own unmet requirements, because
  // "which one" is the only thing that makes either diagnosable.
  const missingEndpoints = useMemo(
    () => missingCreateEndpoints(entry),
    [entry],
  );
  const isUnsupported =
    capabilities.supported === false || missingEndpoints.length > 0;
  const unmet = [...capabilities.unmet, ...missingEndpoints];
  const showPrerequisites =
    prerequisites.blockingIntegrations.length > 0 ||
    prerequisites.warningIntegrations.length > 0;
  // Prerequisites resolve asynchronously; deriving the step keeps an entry with
  // nothing to check from flashing an empty first screen.
  const currentStep: SetupStep =
    step === "prerequisites" && !showPrerequisites ? "form" : step;

  const testRuns = useAutomationRuns({
    id: createdAutomationId ?? "",
    enabled: currentStep === "test" && testRunId !== null,
  });
  const testRun =
    (testRunId
      ? testRuns.data?.runs.find((run) => run.id === testRunId)
      : null) ??
    (dispatchedRun?.id === testRunId ? dispatchedRun : null);
  const testIsRunning = isRunInFlight(testRun);
  const testPassed = testRun?.status === AutomationRunStatus.COMPLETED;
  const isBusy =
    isSubmitting || isDispatchingTest || isFinalizing || testIsRunning;

  const setFieldValue = (name: string, value: SetupFormValue) => {
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
    if (blurTimerRef.current) window.clearTimeout(blurTimerRef.current);
    blurTimerRef.current = window.setTimeout(() => {
      void runPreflight(valuesRef.current).then((result) => {
        if (result) setServiceErrors(result);
      });
    }, PREFLIGHT_DEBOUNCE_MS);
  };

  const handleContinue = async () => {
    if (currentStep === "prerequisites") {
      setStep("form");
      return;
    }

    const failures = validateFormValues(entry.setup, values, overrides);
    if (Object.keys(failures).length > 0) {
      setLocalErrors(failures);
      return;
    }
    setLocalErrors({});

    const result = await runPreflight(values);
    if (result && hasAnyError(result)) {
      setServiceErrors(result);
      return;
    }

    setServiceErrors(NO_SERVICE_ERRORS);
    trackAutomationSetupValidated({ automationId: entry.id });
    setStep("review");
  };

  const submitAction = async (
    actionPayload: SetupRequestBody | null,
    setupMode: SetupMode,
  ) => {
    setIsSubmitting(true);
    try {
      const { response, created } = await runAction(
        entry,
        values,
        actionPayload,
      );
      trackAutomationSetupCreated({
        automationId: entry.id,
        setupMode,
      });

      if (created && typeof response.id === "string") {
        setCreatedAutomationId(response.id);
        setTestRunId(null);
        setDispatchedRun(null);
        setServiceErrors(NO_SERVICE_ERRORS);
        setStep("test");
        return;
      }

      // Assisted setup opens its conversation, and an idempotent direct create
      // opens the already-existing automation without changing or disabling it.
      const destination = getDestination(response);
      if (destination) navigate(destination, { replace: true });
      else onClose();
    } catch (error) {
      trackAutomationSetupFailed({
        automationId: entry.id,
        setupMode,
      });
      const mapped = mapServiceErrors(
        normalizeServiceErrors(getApiErrorBody(error), actionPayload),
        errorMap,
      );
      setServiceErrors(
        hasAnyError(mapped)
          ? mapped
          : {
              fieldErrors: {},
              formErrors: [t(I18nKey.SETUP$SUBMIT_FAILED)],
            },
      );
      setStep("form");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleConfirm = () => submitAction(payload, entry.setup.mode);

  const handleRunTest = async () => {
    if (!createdAutomationId) return;
    setIsDispatchingTest(true);
    setServiceErrors(NO_SERVICE_ERRORS);
    try {
      const run = await AutomationService.dispatchAutomation(
        createdAutomationId,
      );
      setDispatchedRun(run);
      setTestRunId(run.id);
    } catch (error) {
      setServiceErrors({
        fieldErrors: {},
        formErrors: [
          getApiErrorMessage(error, t(I18nKey.AUTOMATIONS$RUN_NOW_ERROR)),
        ],
      });
    } finally {
      setIsDispatchingTest(false);
    }
  };

  const handleEditAfterTest = async () => {
    if (!createdAutomationId || isBusy) return;
    setIsSubmitting(true);
    setServiceErrors(NO_SERVICE_ERRORS);
    try {
      // The setup action's unique pending trigger proved ownership of this
      // disabled record. Removing it before editing is what makes a later
      // create use the changed answers instead of hitting template idempotency.
      await AutomationService.deleteAutomation(createdAutomationId);
      setCreatedAutomationId(null);
      setTestRunId(null);
      setDispatchedRun(null);
      setStep("form");
    } catch (error) {
      setServiceErrors({
        fieldErrors: {},
        formErrors: [getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC))],
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleFinalize = async () => {
    if (!createdAutomationId || !testPassed || isBusy) return;
    setIsFinalizing(true);
    setServiceErrors(NO_SERVICE_ERRORS);
    try {
      await AutomationService.toggleAutomation(createdAutomationId, true);
      navigate(automationDetailPath(createdAutomationId), { replace: true });
    } catch (error) {
      setServiceErrors({
        fieldErrors: {},
        formErrors: [getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC))],
      });
    } finally {
      setIsFinalizing(false);
    }
  };

  const handleClose = async () => {
    if (isBusy) return;
    if (!createdAutomationId) {
      onClose();
      return;
    }

    setIsSubmitting(true);
    setServiceErrors(NO_SERVICE_ERRORS);
    try {
      // Closing setup is abandonment, not finalization. Remove only the record
      // whose unique pending marker proved it belongs to this setup session.
      await AutomationService.deleteAutomation(createdAutomationId);
      onClose();
    } catch (error) {
      setServiceErrors({
        fieldErrors: {},
        formErrors: [getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC))],
      });
      setIsSubmitting(false);
    }
  };

  // A direct entry whose deployment cannot run the direct path degrades to the
  // assisted outcome: the skill command and the entry's fallback message seed
  // a conversation that finishes setup instead.
  const handleFallbackConversation = () => submitAction(null, "assisted");
  const canFallBackToConversation =
    entry.setup.mode === "direct" &&
    (findAutomationCommand(entry) !== null ||
      entry.setup.message !== undefined);

  const resolveFieldError = (name: string): string | undefined => {
    const local = localErrors[name];
    return local ? formatFieldError(local, t) : serviceErrors.fieldErrors[name];
  };

  const isLoading = capabilities.isLoading || prerequisites.isLoading;

  const title = (() => {
    if (isUnsupported) return t(I18nKey.SETUP$UNAVAILABLE_TITLE);
    if (currentStep === "prerequisites")
      return t(I18nKey.SETUP$PREREQUISITES_TITLE);
    if (currentStep === "review") return t(I18nKey.SETUP$REVIEW_TITLE);
    if (currentStep === "test") return t(I18nKey.AUTOMATIONS$RUN_NOW);
    return entry.name;
  })();

  return (
    <ModalBackdrop
      onClose={() => void handleClose()}
      aria-label={entry.name}
    >
      <div
        data-testid="setup-dialog"
        className="relative flex max-h-[85vh] w-[92vw] max-w-lg flex-col rounded-xl border border-[var(--oh-border)] bg-base-secondary"
      >
        <ModalCloseButton
          onClose={() => void handleClose()}
          testId="setup-dialog-close"
          disabled={isBusy}
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
            <div className="flex flex-col gap-2">
              <p className="text-sm text-[var(--oh-muted)]">
                {t(I18nKey.SETUP$UNSUPPORTED_MESSAGE)}
              </p>
              {/* The unmet requirements are the names both sides of the
                  contract already use, so they are printed rather than
                  translated. Without them the block is undiagnosable: the
                  deployment answered, and the host would be discarding the
                  one thing it learned. */}
              {unmet.length > 0 && (
                <p
                  data-testid="setup-unmet-requirements"
                  className="text-sm text-[var(--oh-muted)]"
                >
                  {unmet.join(", ")}
                </p>
              )}
            </div>
          )}

          {!isLoading && !isUnsupported && currentStep === "prerequisites" && (
            <SetupPrerequisitesStep prerequisites={prerequisites} />
          )}

          {!isLoading && !isUnsupported && currentStep === "form" && (
            <div className="flex flex-col gap-5">
              <p className="text-sm text-[var(--oh-muted)]">
                {entry.description}
              </p>
              {entry.setup.form.note && (
                <p className="text-sm text-[var(--oh-muted)]">
                  {entry.setup.form.note}
                </p>
              )}
              {Object.entries(fields).map(([name, field]) => (
                <SetupFormField
                  key={name}
                  name={name}
                  field={field}
                  value={values[name] ?? ""}
                  error={resolveFieldError(name)}
                  options={getFieldOptions(name, field, overrides)}
                  repository={repositories[name] ?? null}
                  disabled={isSubmitting}
                  onChange={(value) => setFieldValue(name, value)}
                  onRepositoryChange={(repository) =>
                    setRepositories((current) => ({
                      ...current,
                      [name]: repository,
                    }))
                  }
                  onBlur={handleFieldBlur}
                />
              ))}
            </div>
          )}

          {!isLoading && !isUnsupported && currentStep === "review" && (
            <SetupReviewStep setup={entry.setup} values={values} />
          )}

          {!isLoading && !isUnsupported && currentStep === "test" && (
            <SetupTestRunStep run={testRun} description={entry.description} />
          )}

          {serviceErrors.formErrors.map((message) => (
            <p
              key={message}
              role="alert"
              data-testid="setup-form-error"
              className="pt-4 text-sm text-red-400"
            >
              {message}
            </p>
          ))}
        </div>

        <footer className="flex flex-shrink-0 justify-end gap-2 px-6 pb-6 pt-4">
          {(currentStep === "review" || currentStep === "test") && (
            <BrandButton
              testId="setup-back-button"
              type="button"
              variant="secondary"
              isDisabled={isBusy}
              onClick={
                currentStep === "test"
                  ? handleEditAfterTest
                  : () => setStep("form")
              }
            >
              {t(I18nKey.BUTTON$BACK)}
            </BrandButton>
          )}

          {isUnsupported ? (
            <>
              <BrandButton
                type="button"
                variant="secondary"
                isDisabled={isSubmitting}
                onClick={onClose}
              >
                {t(I18nKey.BUTTON$CLOSE)}
              </BrandButton>
              {canFallBackToConversation && (
                <BrandButton
                  testId="setup-fallback-conversation"
                  type="button"
                  variant="primary"
                  isDisabled={isSubmitting}
                  onClick={handleFallbackConversation}
                >
                  {t(I18nKey.SETUP$FALLBACK_CONVERSATION)}
                </BrandButton>
              )}
            </>
          ) : currentStep === "test" ? (
            testPassed ? (
              <BrandButton
                testId="setup-finalize-button"
                type="button"
                variant="primary"
                isDisabled={isBusy}
                onClick={handleFinalize}
              >
                {t(I18nKey.AUTOMATIONS$TURN_ON)}
              </BrandButton>
            ) : (
              <BrandButton
                testId="setup-test-run-button"
                type="button"
                variant="primary"
                isDisabled={isBusy}
                onClick={handleRunTest}
              >
                {testIsRunning
                  ? t(I18nKey.AUTOMATIONS$DETAIL$RUNNING)
                  : isDispatchingTest
                    ? t(I18nKey.AUTOMATIONS$STARTING)
                    : t(I18nKey.AUTOMATIONS$RUN_NOW)}
              </BrandButton>
            )
          ) : (
            <BrandButton
              testId="setup-continue-button"
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
                ? t(I18nKey.SETUP$CONFIRM)
                : t(I18nKey.BUTTON$CONTINUE)}
            </BrandButton>
          )}
        </footer>
      </div>
    </ModalBackdrop>
  );
}

function formatFieldError(
  error: SetupFieldError,
  t: (key: I18nKey, options?: Record<string, unknown>) => string,
): string {
  switch (error.code) {
    case "required":
      return t(I18nKey.SETUP$VALIDATION_REQUIRED);
    case "minLength":
      return t(I18nKey.SETUP$VALIDATION_MIN_LENGTH, { length: error.length });
    case "maxLength":
      return t(I18nKey.SETUP$VALIDATION_MAX_LENGTH, { length: error.length });
    case "invalidOption":
      return t(I18nKey.SETUP$VALIDATION_INVALID_OPTION);
    case "unsafeExpressionLiteral":
      return t(I18nKey.SETUP$VALIDATION_UNSAFE_VALUE);
    default: {
      // A code added to `SetupFieldError` without a message here is a compile
      // error rather than a field that silently reads "this field is required".
      const exhaustive: never = error;
      return exhaustive;
    }
  }
}
