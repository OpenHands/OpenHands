import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { ArrowUpRight, Check, ClipboardList, X } from "lucide-react";
import { getLockedCloudHost } from "#/api/agent-server-config";
import { NavigationLink } from "#/components/shared/navigation-link";
import { useNavigation } from "#/context/navigation-context";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { SUPER_ADMIN_SETUP_GUIDE_PAGE_PATH } from "./super-admin-setup-guide.constants";
import { useSuperAdminSetupGuide } from "./use-super-admin-setup-guide";

const ICON_BUTTON_CLASS = cn(
  "inline-flex size-7 shrink-0 items-center justify-center rounded-md",
  "text-muted hover:bg-surface hover:text-contrast",
);

const STEP_ROW_CLASS =
  "flex w-full min-w-0 items-center gap-2.5 rounded-md px-2.5 py-1.5 hover:bg-surface";

/**
 * Floating lower-right setup guide for the enterprise Super Admin, matching
 * the guide in the OpenHands Enterprise app. Steps that live in the
 * enterprise app link back to it; progress is read from the server.
 */
export default function SuperAdminSetupGuide() {
  const { t } = useTranslation("openhands");
  const { currentPath } = useNavigation();
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

  // Progress is read from the server, so re-read it as the admin moves around.
  useEffect(() => {
    if (visible) {
      refetch();
    }
  }, [currentPath, visible, refetch]);

  if (!visible) {
    return null;
  }

  const cloudHost = backend.host.replace(/\/+$/, "");
  // Locked-to-Cloud serves the canvas on the cloud host itself, so enterprise
  // pages open in this tab; standalone / Electron keep a new tab.
  const isLockedToCloud = getLockedCloudHost() !== null;
  const cloudLinkProps = {
    target: isLockedToCloud ? undefined : "_blank",
    rel: isLockedToCloud ? undefined : "noopener noreferrer",
  };
  const cloudUrl = (path: string, withOrg: boolean) =>
    withOrg && guideOrgId
      ? `${cloudHost}${path}?org=${encodeURIComponent(guideOrgId)}`
      : `${cloudHost}${path}`;

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
                className="h-full rounded-full bg-primary transition-all duration-300"
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
                        ? "border-primary bg-primary text-on-primary"
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
                  {step.destination.kind === "canvas" ? (
                    <NavigationLink
                      to={step.destination.path}
                      data-testid={testId}
                      className={rowClassName}
                    >
                      {label}
                    </NavigationLink>
                  ) : (
                    <a
                      href={cloudUrl(
                        step.destination.path,
                        step.destination.withOrg,
                      )}
                      {...cloudLinkProps}
                      data-testid={testId}
                      className={rowClassName}
                    >
                      {label}
                    </a>
                  )}
                </li>
              );
            })}
          </ul>

          {nextStep ? (
            <p
              className="text-xs text-muted"
              data-testid="super-admin-setup-guide-next"
            >
              {t(I18nKey.ONBOARDING$SETUP_GUIDE_NEXT, {
                step: t(nextStep.labelKey),
              })}
            </p>
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
