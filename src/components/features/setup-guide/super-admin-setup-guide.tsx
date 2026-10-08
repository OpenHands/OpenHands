import {
  type MouseEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { useTranslation } from "react-i18next";
import { ArrowUpRight, Check, ClipboardList, X } from "lucide-react";
import { getLockedCloudHost } from "#/api/agent-server-config";
import { NavigationLink } from "#/components/shared/navigation-link";
import { useNavigation } from "#/context/navigation-context";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { I18nKey } from "#/i18n/declaration";
import { formControlButtonClassName } from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";
import {
  SETUP_TOUR_PARAM,
  SUPER_ADMIN_SETUP_GUIDE_PAGE_PATH,
  SUPER_ADMIN_SETUP_STEPS,
  type SuperAdminSetupStep,
  type SuperAdminSetupStepId,
} from "./super-admin-setup-guide.constants";
import { SUPER_ADMIN_SETUP_STEP_EVENT } from "./super-admin-setup-step-event";
import {
  hasSetupGuideTour,
  startSetupGuideTour,
} from "./tour/setup-guide-tour";
import {
  isGuidedTourActive,
  subscribeGuidedTourActive,
} from "./tour/tour-engine";
import {
  getNextSetupStep,
  useSuperAdminSetupGuide,
} from "./use-super-admin-setup-guide";

const ICON_BUTTON_CLASS = cn(
  "inline-flex size-7 shrink-0 items-center justify-center rounded-md",
  "text-muted hover:bg-surface hover:text-contrast",
);

const STEP_ROW_CLASS =
  "flex w-full min-w-0 items-center gap-2.5 rounded-md px-2.5 py-1.5 hover:bg-surface";

// The enterprise guide's primary button: light on dark. Canvas's `primary`
// is its gold accent, so this uses `foreground` like the enterprise app.
const START_BUTTON_CLASS = cn(
  formControlButtonClassName,
  "w-full bg-foreground font-medium text-on-primary hover:opacity-80",
);

const START_TEST_ID = "super-admin-setup-guide-start";

/**
 * A page on the cloud host, on the guide's organization when given, and
 * naming the step whose tour the enterprise guide should start.
 */
function cloudPageUrl(
  cloudHost: string,
  path: string,
  orgId: string | null,
  tourStepId?: SuperAdminSetupStepId,
) {
  const params = new URLSearchParams();
  if (orgId) params.set("org", orgId);
  if (tourStepId) params.set(SETUP_TOUR_PARAM, tourStepId);
  const query = params.toString();
  return query ? `${cloudHost}${path}?${query}` : `${cloudHost}${path}`;
}

/**
 * Floating lower-right setup guide for the enterprise Super Admin, matching
 * the guide in the OpenHands Enterprise app. Steps that live in the
 * enterprise app link back to it; progress is read from the server.
 */
export default function SuperAdminSetupGuide() {
  const { t } = useTranslation("openhands");
  const { currentPath, navigate } = useNavigation();
  const { backend } = useActiveBackend();
  const {
    steps,
    completedCount,
    totalCount,
    nextStep,
    guideOrgId,
    visible,
    refetch,
  } = useSuperAdminSetupGuide();
  const [open, setOpen] = useState(true);
  const tourActive = useSyncExternalStore(
    subscribeGuidedTourActive,
    isGuidedTourActive,
    isGuidedTourActive,
  );

  // The tour checks the current route after it navigates, between renders.
  const pathRef = useRef(currentPath);
  useEffect(() => {
    pathRef.current = currentPath;
  }, [currentPath]);
  const startTour = useCallback(
    (stepId: SuperAdminSetupStepId) =>
      startSetupGuideTour(
        stepId,
        { navigate, getPath: () => pathRef.current },
        t,
      ),
    [navigate, t],
  );

  // Progress is read from the server, so re-read it as the admin moves around.
  useEffect(() => {
    if (visible) {
      refetch();
    }
  }, [currentPath, visible, refetch]);

  const cloudHost = backend.host.replace(/\/+$/, "");
  // Locked-to-Cloud serves the canvas on the cloud host itself, so enterprise
  // pages open in this tab; standalone / Electron keep a new tab.
  const isLockedToCloud = getLockedCloudHost() !== null;

  // When the guide's next step is done, open the step after it the way Start
  // does. A step finished out of order, or not confirmed by the server, only
  // refreshes the progress; nothing opens after the last required step.
  const nextStepId = nextStep?.id ?? null;
  useEffect(() => {
    if (!visible) {
      return undefined;
    }
    const onStep = async (event: Event) => {
      const completedId = (event as CustomEvent<{ id?: string }>).detail?.id;
      const { data } = await refetch();
      const guideSteps = data?.guide_steps;
      if (!guideSteps || !completedId || completedId !== nextStepId) {
        return;
      }
      const after = getNextSetupStep(guideSteps);
      if (!after || after.id === completedId) {
        return;
      }
      const { destination } = after;
      if (destination.kind === "canvas") {
        if (hasSetupGuideTour(after.id)) {
          await startTour(after.id);
        } else {
          navigate(destination.path);
        }
      } else if (isLockedToCloud) {
        window.location.assign(
          cloudPageUrl(
            cloudHost,
            destination.path,
            destination.withOrg ? guideOrgId : null,
            after.id,
          ),
        );
      } else {
        // A new tab opened after a save is blocked, so point at Start instead.
        setOpen(true);
      }
    };
    window.addEventListener(SUPER_ADMIN_SETUP_STEP_EVENT, onStep);
    return () =>
      window.removeEventListener(SUPER_ADMIN_SETUP_STEP_EVENT, onStep);
  }, [
    visible,
    refetch,
    nextStepId,
    navigate,
    startTour,
    cloudHost,
    guideOrgId,
    isLockedToCloud,
  ]);

  // A page opened from the enterprise guide names the step whose tour to
  // start. Enterprise steps have no tour here, so only Canvas steps start.
  useEffect(() => {
    if (!visible) {
      return;
    }
    const params = new URLSearchParams(window.location.search);
    const requested = params.get(SETUP_TOUR_PARAM);
    if (!requested) {
      return;
    }
    // Drop it first, so the tour starts once.
    params.delete(SETUP_TOUR_PARAM);
    const query = params.toString();
    window.history.replaceState(
      window.history.state,
      "",
      `${window.location.pathname}${query ? `?${query}` : ""}${window.location.hash}`,
    );
    const step = SUPER_ADMIN_SETUP_STEPS.find(({ id }) => id === requested);
    if (step && hasSetupGuideTour(step.id)) {
      setOpen(false);
      startTour(step.id);
    }
  }, [visible, startTour]);

  if (!visible || tourActive) {
    return null;
  }

  const cloudLinkProps = {
    target: isLockedToCloud ? undefined : "_blank",
    rel: isLockedToCloud ? undefined : "noopener noreferrer",
  };
  const cloudUrl = (
    path: string,
    withOrg: boolean,
    tourStepId?: SuperAdminSetupStepId,
  ) => cloudPageUrl(cloudHost, path, withOrg ? guideOrgId : null, tourStepId);
  // A step row and Start open the same page for a step; Start also asks the
  // enterprise guide for the step's tour.
  const renderStepLink = (
    step: SuperAdminSetupStep,
    testId: string,
    className: string,
    children: ReactNode,
    onClick?: (event: MouseEvent<HTMLAnchorElement>) => void,
    startsTour = false,
  ) =>
    step.destination.kind === "canvas" ? (
      <NavigationLink
        to={step.destination.path}
        data-testid={testId}
        className={className}
        onClick={onClick}
      >
        {children}
      </NavigationLink>
    ) : (
      <a
        href={cloudUrl(
          step.destination.path,
          step.destination.withOrg,
          startsTour ? step.id : undefined,
        )}
        {...cloudLinkProps}
        data-testid={testId}
        className={className}
        onClick={onClick}
      >
        {children}
      </a>
    );

  const title = t(I18nKey.ONBOARDING$SETUP_GUIDE_TITLE);
  const progressPct = Math.round((completedCount / totalCount) * 100);

  return (
    <div
      className="fixed bottom-5 right-5 z-30 flex flex-col items-end gap-2"
      data-testid="super-admin-setup-guide"
    >
      {open ? (
        <div
          data-testid="super-admin-setup-guide-panel"
          className={cn(
            "flex w-80 flex-col gap-3 rounded-xl border border-border",
            "bg-surface-raised p-4 shadow-lg",
          )}
        >
          <div className="flex items-center justify-between gap-2">
            <p className="min-w-0 text-sm font-semibold text-content">
              {title}
            </p>
            <div className="flex shrink-0 items-center gap-0.5">
              <a
                href={cloudUrl(SUPER_ADMIN_SETUP_GUIDE_PAGE_PATH, false)}
                {...cloudLinkProps}
                data-testid="super-admin-setup-guide-page"
                aria-label={title}
                title={title}
                className={ICON_BUTTON_CLASS}
              >
                <ArrowUpRight className="size-4" strokeWidth={2} aria-hidden />
              </a>
              <button
                type="button"
                aria-label={t(I18nKey.BUTTON$CLOSE)}
                className={ICON_BUTTON_CLASS}
                onClick={() => setOpen(false)}
              >
                <X className="size-4" strokeWidth={2} aria-hidden />
              </button>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div
              className="h-1.5 flex-1 overflow-hidden rounded-full bg-border"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={progressPct}
              aria-label={title}
            >
              <div
                className="h-full rounded-full bg-foreground transition-all duration-300"
                style={{ width: `${progressPct}%` }}
              />
            </div>
            <span className="shrink-0 text-xs tabular-nums text-muted">
              {completedCount}/{totalCount}
            </span>
          </div>

          <ul className="flex flex-col gap-0.5">
            {steps.map((step) => {
              const label = (
                <>
                  <span
                    aria-hidden
                    className={cn(
                      "inline-flex size-4 shrink-0 items-center justify-center rounded-full border",
                      step.done
                        ? "border-foreground bg-foreground text-on-primary"
                        : "border-border bg-transparent",
                    )}
                  >
                    {step.done ? (
                      <Check className="size-2.5" strokeWidth={3} />
                    ) : null}
                  </span>
                  <span
                    className={cn(
                      "min-w-0 flex-1 truncate text-xs",
                      step.done ? "text-muted line-through" : "text-content",
                    )}
                  >
                    {t(step.labelKey)}
                  </span>
                </>
              );
              const testId = `super-admin-setup-guide-step-${step.id}`;
              const rowClassName = cn(
                STEP_ROW_CLASS,
                nextStep?.id === step.id && "bg-surface",
              );
              return (
                <li key={step.id}>
                  {renderStepLink(step, testId, rowClassName, label)}
                </li>
              );
            })}
          </ul>

          {nextStep ? (
            <div className="flex flex-col gap-2">
              <p
                className="text-xs text-muted"
                data-testid="super-admin-setup-guide-next"
              >
                {t(I18nKey.ONBOARDING$SETUP_GUIDE_NEXT, {
                  step: t(nextStep.labelKey),
                })}
              </p>
              {/* Like the enterprise Start, close the panel before opening
                  the step, so Start does something even when the admin is
                  already on that step's page. A Canvas step's tour opens
                  the page itself; a modified click still follows the link. */}
              {renderStepLink(
                nextStep,
                START_TEST_ID,
                START_BUTTON_CLASS,
                t(I18nKey.ONBOARDING$SETUP_GUIDE_START),
                (event) => {
                  setOpen(false);
                  const plainClick =
                    event.button === 0 &&
                    !event.metaKey &&
                    !event.ctrlKey &&
                    !event.shiftKey &&
                    !event.altKey;
                  if (
                    plainClick &&
                    nextStep.destination.kind === "canvas" &&
                    hasSetupGuideTour(nextStep.id)
                  ) {
                    event.preventDefault();
                    startTour(nextStep.id);
                  }
                },
                true,
              )}
            </div>
          ) : null}
        </div>
      ) : null}

      <button
        type="button"
        data-testid="super-admin-setup-guide-toggle"
        aria-label={title}
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
        className={cn(
          "flex items-center gap-2 rounded-full border border-border",
          "bg-surface-raised px-3.5 py-2 text-sm font-medium text-content",
          "shadow-sm hover:bg-surface",
        )}
      >
        <ClipboardList className="size-4" strokeWidth={2} aria-hidden />
        {title}
      </button>
    </div>
  );
}
