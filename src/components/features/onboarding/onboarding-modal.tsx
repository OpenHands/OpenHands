import React from "react";
import { useTranslation } from "react-i18next";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { getLockedCloudHost, isSameCloudHost } from "#/api/agent-server-config";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import {
  MODAL_MAX_WIDTH_VIEWPORT,
  modalWidthClassName,
} from "#/components/shared/modals/modal-body";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import { useBackendsHealth } from "#/hooks/query/use-backends-health";
import { useSettings } from "#/hooks/query/use-settings";
import { useTracking } from "#/hooks/use-tracking";
import { OnboardingProgressBar } from "./onboarding-progress-bar";
import {
  ChooseAgentStep,
  type OnboardingAgentId,
} from "./steps/choose-agent-step";
import { CheckBackendStep } from "./steps/check-backend-step";
import { SetupLlmStep } from "./steps/setup-llm-step";
import { SetupAcpSecretsStep } from "./steps/setup-acp-secrets-step";
import { SayHelloStep } from "./steps/say-hello-step";

/**
 * Logical onboarding phases.
 *
 * State is tracked as a phase (not a numeric step) so that the slide
 * indices renumbering when ``skipBackendStep`` flips — e.g. a returning
 * Cloud user finishes the backend slide and the carousel collapses from
 * 4 slides to 3 — cannot accidentally drag the user from "agent" onto
 * the slide that used to live at the agent index ("setup"). See the
 * regression test `onboarding-modal` covering this transition.
 */
type OnboardingPhase = "backend" | "agent" | "setup" | "hello";

const PHASE_ORDER_WITH_BACKEND: readonly OnboardingPhase[] = [
  "backend",
  "agent",
  "setup",
  "hello",
];
const PHASE_ORDER_WITHOUT_BACKEND: readonly OnboardingPhase[] = [
  "agent",
  "setup",
  "hello",
];

/**
 * sessionStorage flag marking that the one-time "onboarding started" analytics
 * event has already been captured for this browser session. It survives
 * rerenders, remounts and the lazy/Suspense flip so the event fires at most
 * once per onboarding session.
 */
const ONBOARDING_STARTED_TRACKED_KEY = "openhands-onboarding-started";

interface SlideProps {
  /** Index of this slide in the step sequence. */
  index: number;
  /** Index of the currently visible step. */
  currentStep: number;
  children: React.ReactNode;
}

/**
 * One step panel inside the slide rail.
 *
 * Only the active slide is in normal flow — it drives the surrounding
 * container's height. Inactive slides are absolutely positioned so
 * they don't add their height to the modal box (which previously made
 * the modal "overhang" with empty space sized to the tallest step).
 *
 * Each slide is translated horizontally by `(index - currentStep) *
 * 100%` so the active step sits at offset 0, with prior steps off to
 * the left and upcoming steps off to the right. Changes to
 * `currentStep` smoothly animate the transform.
 */
function Slide({ index, currentStep, children }: SlideProps) {
  const isActive = index === currentStep;
  const offsetPct = (index - currentStep) * 100;
  return (
    <div
      data-testid={`onboarding-slide-${index}`}
      data-active={isActive}
      aria-hidden={!isActive}
      // slide offset computed from step index at runtime
      style={{ transform: `translateX(${offsetPct}%)` }}
      className={cn(
        "w-full transition-transform duration-300 ease-out",
        // Inactive slides are taken out of flow so the rail's height
        // tracks just the active step; they stay overlaid via inset-0
        // so they slide in/out of view across the same horizontal box.
        !isActive && "pointer-events-none absolute inset-0",
      )}
    >
      {children}
    </div>
  );
}

interface OnboardingModalProps {
  /** Called when the user dismisses the modal (skip / X / launch). */
  onClose: () => void;
  /** Optional slide index for dev preview (`?previewOnboardingStep=`). */
  initialStep?: number;
  /** When true, skip/close does not persist onboarding completion. */
  isPreview?: boolean;
}

/**
 * Top-level onboarding modal for first-time users.
 *
 * The flow starts with backend setup only when the active backend is missing
 * or cannot be reached. If an already configured backend is healthy, the user
 * starts directly on agent selection:
 *   0. Check/add backend (only when needed)
 *   1. Choose agent
 *   2. Set up LLM
 *   3. Say hello (creates a fresh conversation, then closes)
 *
 * Internally we track the user's *phase* — "backend" | "agent" | "setup" |
 * "hello" — rather than a numeric step, because the slide indices renumber
 * when the backend slide is skipped and we must never accidentally drop the
 * user onto whatever slide *used to* live at their numeric position.
 *
 * Each visible step lives in its own slide and the rail is translated
 * horizontally by step index, so transitioning between steps animates the new
 * step in from the right.
 */
export function OnboardingModal({
  onClose,
  initialStep = 0,
  isPreview = false,
}: OnboardingModalProps) {
  const { t } = useTranslation("openhands");
  const { data: settings } = useSettings();
  const analyticsEnabled = settings?.user_consents_to_analytics === true;
  const {
    trackOnboardingStarted,
    trackOnboardingStepViewed,
    trackOnboardingCompleted,
    trackOnboardingSkipped,
  } = useTracking();
  const { active } = useActiveBackendContext();
  const { backend } = active;
  const noBackendSelected = isNoBackend(backend);
  const healthByBackendId = useBackendsHealth(
    noBackendSelected ? [] : [backend],
  );
  // In locked-to-Cloud mode the backend slide may only be skipped when the
  // active backend IS the configured locked Cloud host. A reachable stale
  // Local backend (or a Cloud backend on a different host) must keep
  // `CheckBackendStep` visible so the user can log into Cloud and replace
  // the stale backend — otherwise they would continue as Local despite
  // `VITE_LOCK_TO_CLOUD`. Outside locked mode the existing behavior
  // (skip once the active backend is healthy) is unchanged.
  const lockedCloudHost = getLockedCloudHost();
  const isActiveLockedCloudBackend =
    lockedCloudHost !== null &&
    backend.kind === "cloud" &&
    isSameCloudHost(backend.host, lockedCloudHost);
  const skipBackendStep =
    !noBackendSelected &&
    healthByBackendId[backend.id]?.isConnected === true &&
    (lockedCloudHost === null || isActiveLockedCloudBackend);

  const slideOrder = skipBackendStep
    ? PHASE_ORDER_WITHOUT_BACKEND
    : PHASE_ORDER_WITH_BACKEND;

  const [phase, setPhase] = React.useState<OnboardingPhase>(
    () =>
      PHASE_ORDER_WITH_BACKEND[
        Math.min(Math.max(initialStep, 0), PHASE_ORDER_WITH_BACKEND.length - 1)
      ],
  );
  const [selectedAgentId, setSelectedAgentId] =
    React.useState<OnboardingAgentId>("openhands");

  // When the backend slide drops out of the flow (skipBackendStep flips
  // true), a user still parked on "backend" must be moved forward to the
  // first remaining phase. Doing this with the *phase* rather than a numeric
  // step keeps every other phase pinned to itself, so the user never lands
  // on the slide that used to live at the next index ("setup").
  React.useEffect(() => {
    if (!slideOrder.includes(phase)) setPhase(slideOrder[0]);
  }, [phase, slideOrder]);

  const totalSteps = slideOrder.length;
  const currentPhase = slideOrder.includes(phase) ? phase : slideOrder[0];
  const currentStep = slideOrder.indexOf(currentPhase);

  // Backend connectivity is "settled" once we know whether the active backend
  // is reachable (or no backend is selected). Until then `skipBackendStep` may
  // still flip true and renumber the slides, briefly showing the backend slide
  // for an already-healthy backend. Gate Step Viewed on this so we never emit a
  // phantom "backend" view that immediately auto-skips to "agent".
  const backendHealthSettled =
    noBackendSelected ||
    healthByBackendId[backend.id]?.isConnected !== undefined;

  const isOpenHands = selectedAgentId === "openhands";
  const hideSkip = currentStep === 0 && getLockedCloudHost() !== null;
  const goNext = React.useCallback(() => {
    setPhase((prev) => {
      const order = slideOrder;
      const idx = order.indexOf(prev);
      // If `prev` isn't in the current order (e.g. "backend" after the
      // backend slide just collapsed), start from the first slot.
      const safeIdx = idx === -1 ? 0 : idx;
      return order[Math.min(safeIdx + 1, order.length - 1)];
    });
  }, [slideOrder]);
  const goBack = React.useCallback(() => {
    setPhase((prev) => {
      const order = slideOrder;
      const idx = order.indexOf(prev);
      const safeIdx = idx === -1 ? 0 : idx;
      return order[Math.max(safeIdx - 1, 0)];
    });
  }, [slideOrder]);

  // --- Onboarding analytics -------------------------------------------------
  // Every event is routed through `useTracking`, which drops captures unless
  // the user has consented; the design-preview harness (`isPreview`) emits
  // nothing.

  // Onboarding Started — fire at most once per browser session. The
  // sessionStorage flag survives rerenders, remounts and the lazy/Suspense
  // flip; the consent gate means we only arm the guard once a capture can
  // actually happen, so a user who consents partway through onboarding still
  // gets the event instead of having it latched away during the no-consent
  // window.
  const startedTrackedRef = React.useRef(false);
  React.useEffect(() => {
    if (isPreview || !analyticsEnabled || startedTrackedRef.current) return;
    if (window.sessionStorage.getItem(ONBOARDING_STARTED_TRACKED_KEY)) return;
    startedTrackedRef.current = true;
    window.sessionStorage.setItem(ONBOARDING_STARTED_TRACKED_KEY, "1");
    trackOnboardingStarted();
  }, [isPreview, analyticsEnabled]);

  // Onboarding Step Viewed — fire once per phase entry. The phase string is the
  // stable identity (slide indices renumber), so dedupe on it: this absorbs
  // StrictMode's double-invoke and the index renumber, while re-entering a step
  // via Back counts as a genuine new view.
  const lastViewedPhaseRef = React.useRef<OnboardingPhase | null>(null);
  React.useEffect(() => {
    if (isPreview || !analyticsEnabled || !backendHealthSettled) return;
    if (lastViewedPhaseRef.current === currentPhase) return;
    lastViewedPhaseRef.current = currentPhase;
    trackOnboardingStepViewed({
      step: currentPhase,
      stepIndex: currentStep,
      totalSteps,
      agent: selectedAgentId,
    });
  }, [
    isPreview,
    analyticsEnabled,
    backendHealthSettled,
    currentPhase,
    currentStep,
    totalSteps,
    selectedAgentId,
  ]);

  // Onboarding Completed — `onLaunched` runs only after a conversation is
  // successfully created (the hello message or a recommended automation), so
  // wrapping it captures completion exactly once before the modal closes.
  const handleCompleted = () => {
    trackOnboardingCompleted({ agent: selectedAgentId });
    onClose();
  };

  // Onboarding Skipped/Dismissed — the user exited before completing, via the
  // skip button (non-final steps) or the final-step Close button. `currentPhase`
  // records where they left, which also distinguishes the two affordances.
  const handleSkipOrDismiss = () => {
    trackOnboardingSkipped({
      step: currentPhase,
      stepIndex: currentStep,
      totalSteps,
      agent: selectedAgentId,
    });
    onClose();
  };

  return (
    // No `onClose`: the flow must only be dismissed via explicit actions
    // (the skip button or launching), never by an errant backdrop click or
    // Escape press — see https://github.com/OpenHands/agent-canvas/issues/1085.
    <ModalBackdrop aria-label={t(I18nKey.ONBOARDING$TITLE)}>
      <div className="relative flex flex-col items-center gap-4">
        <section
          data-testid="onboarding-modal"
          data-current-step={currentStep}
          data-preview={isPreview ? "true" : undefined}
          className={cn(
            "flex flex-col gap-6 overflow-hidden rounded-2xl border border-white/10 bg-base-secondary shadow-2xl",
            modalWidthClassName("lg"),
            MODAL_MAX_WIDTH_VIEWPORT,
            "max-h-[90vh]",
          )}
        >
          <header className="flex flex-col gap-3 px-7 pt-7 shrink-0">
            <OnboardingProgressBar
              currentStep={currentStep}
              totalSteps={totalSteps}
            />
          </header>

          <div
            data-testid="onboarding-scroll-area"
            className="flex-1 min-h-0 overflow-y-auto custom-scrollbar-always px-7"
          >
            <div
              data-testid="onboarding-slide-rail"
              data-current-step={currentStep}
              className="relative overflow-clip"
            >
              {skipBackendStep ? null : (
                <Slide
                  index={slideOrder.indexOf("backend")}
                  currentStep={currentStep}
                >
                  <CheckBackendStep onNext={goNext} onClose={onClose} />
                </Slide>
              )}
              <Slide
                index={slideOrder.indexOf("agent")}
                currentStep={currentStep}
              >
                <ChooseAgentStep
                  selectedAgentId={selectedAgentId}
                  onSelect={setSelectedAgentId}
                  onBack={skipBackendStep ? undefined : goBack}
                  onNext={goNext}
                />
              </Slide>
              <Slide
                index={slideOrder.indexOf("setup")}
                currentStep={currentStep}
              >
                {isOpenHands ? (
                  <SetupLlmStep onBack={goBack} onNext={goNext} />
                ) : (
                  <SetupAcpSecretsStep
                    providerKey={selectedAgentId}
                    isActive={currentPhase === "setup"}
                    onBack={goBack}
                    onNext={goNext}
                  />
                )}
              </Slide>
              <Slide
                index={slideOrder.indexOf("hello")}
                currentStep={currentStep}
              >
                <SayHelloStep
                  onBack={goBack}
                  onClose={handleSkipOrDismiss}
                  onLaunched={handleCompleted}
                />
              </Slide>
            </div>
          </div>
        </section>

        {currentStep < totalSteps - 1 && !hideSkip ? (
          <button
            type="button"
            data-testid="onboarding-skip"
            onClick={handleSkipOrDismiss}
            className="rounded-md px-3 py-2 text-sm text-[var(--oh-muted)] transition-colors hover:bg-white/5 hover:text-white cursor-pointer"
          >
            {t(I18nKey.ONBOARDING$SKIP)}
          </button>
        ) : null}
      </div>
    </ModalBackdrop>
  );
}
