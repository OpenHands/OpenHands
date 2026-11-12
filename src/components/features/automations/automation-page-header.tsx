import React from "react";
import { ArrowLeft } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { NavigationLink } from "#/components/shared/navigation-link";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";

interface AutomationPageHeaderProps {
  title: React.ReactNode;
  /** Where the back arrow left of the title goes. */
  backTo: string;
  /** Shown beside the title, e.g. the status badge. */
  badge?: React.ReactNode;
  /** Right-hand actions. */
  children?: React.ReactNode;
  /** The actions row, for callers that portal their actions into it. */
  actionsRef?: React.Ref<HTMLDivElement>;
  testId?: string;
}

/**
 * The top bar shared by an automation's page and its setup form, so viewing,
 * editing and creating one keep the title and actions in the same place.
 */
export function AutomationPageHeader({
  title,
  backTo,
  badge,
  children,
  actionsRef,
  testId,
}: AutomationPageHeaderProps) {
  const { t } = useTranslation("openhands");
  const backLabel = t(I18nKey.COMMON$BACK);

  return (
    <header
      data-testid={testId}
      className="flex min-h-14 shrink-0 items-center border-b border-border bg-canvas-base px-5 py-2"
    >
      {/* Same column as the page body, so the title sits over the content. */}
      <div className="automation-setup-content mx-auto flex w-full min-w-0 flex-wrap items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <StyledTooltip content={backLabel} placement="bottom">
            <NavigationLink
              to={backTo}
              aria-label={backLabel}
              data-testid="automation-page-back"
              className="-ml-1.5 inline-flex size-7 shrink-0 items-center justify-center rounded-md text-muted transition-colors hover:bg-surface-raised hover:text-content"
            >
              <ArrowLeft className="size-4" aria-hidden />
            </NavigationLink>
          </StyledTooltip>
          <h1 className="min-w-0 truncate text-lg font-medium text-content">
            {title}
          </h1>
          {badge}
        </div>
        <div
          ref={actionsRef}
          data-testid={testId ? `${testId}-actions` : undefined}
          className="flex min-w-0 max-w-full flex-wrap items-center gap-1.5"
        >
          {children}
        </div>
      </div>
    </header>
  );
}
