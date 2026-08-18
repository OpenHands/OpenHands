import { useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import ChevronDownSmallIcon from "#/icons/chevron-down-small.svg?react";

interface AdvancedDisclosureProps {
  children: React.ReactNode;
  testId?: string;
}

/**
 * Collapsible section for less-common fields, so the default view stays
 * scannable. Mirrors the disclosure in backend-form-modal.tsx: height
 * animates through `grid-template-rows` (0fr ↔ 1fr) so no max-height guess
 * is needed, content stays mounted while collapsed, and `inert` keeps it out
 * of the tab order.
 */
export function AdvancedDisclosure({
  children,
  testId = "advanced-disclosure",
}: AdvancedDisclosureProps) {
  const { t } = useTranslation("openhands");
  const [open, setOpen] = useState(false);
  const panelId = `${testId}-panel`;

  return (
    <div className="col-span-2 border-t border-[var(--oh-border)] pt-4">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-controls={panelId}
        data-testid={`${testId}-toggle`}
        className="flex items-center gap-1 text-xs text-muted transition-colors hover:text-content-2"
      >
        <span>{t(I18nKey.AUTOMATIONS$DETAIL$ADVANCED)}</span>
        <ChevronDownSmallIcon
          className={cn(
            "h-4 w-4 shrink-0 transition-transform duration-200 ease-out",
            open && "rotate-180",
          )}
          aria-hidden
        />
      </button>
      <div
        className={cn(
          "grid transition-[grid-template-rows] duration-200 ease-out motion-reduce:transition-none",
          open ? "grid-rows-[1fr]" : "grid-rows-[0fr]",
        )}
      >
        <div className="overflow-hidden">
          <div
            id={panelId}
            data-testid={testId}
            aria-hidden={!open}
            inert={!open ? true : undefined}
            className={cn(
              "grid grid-cols-2 gap-x-4 gap-y-5 pt-4 transition-opacity duration-200 ease-out motion-reduce:transition-none",
              open ? "opacity-100" : "opacity-0",
            )}
          >
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}
