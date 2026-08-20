import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import TerminalIcon from "#/icons/terminal.svg?react";
import { cn } from "#/utils/utils";
import { SectionCard } from "./section-card";

/** ~10 lines at `leading-6` (24px). */
const PROMPT_COLLAPSED_MAX_HEIGHT_PX = 240;
const PROMPT_COLLAPSED_MAX_HEIGHT_CLASS = "max-h-60";

interface PromptSectionProps {
  prompt: string;
}

export function PromptSection({ prompt }: PromptSectionProps) {
  const { t } = useTranslation("openhands");
  const [isExpanded, setIsExpanded] = React.useState(false);
  const [isOverflowing, setIsOverflowing] = React.useState(false);
  const contentRef = React.useRef<HTMLParagraphElement>(null);

  const isCollapsed = isOverflowing && !isExpanded;

  React.useLayoutEffect(() => {
    const content = contentRef.current;
    if (!content) {
      return undefined;
    }

    const updateOverflow = () => {
      setIsOverflowing(content.scrollHeight > PROMPT_COLLAPSED_MAX_HEIGHT_PX);
    };

    updateOverflow();

    const observer = new ResizeObserver(updateOverflow);
    observer.observe(content);

    return () => observer.disconnect();
  }, [prompt]);

  return (
    <SectionCard
      icon={<TerminalIcon className="size-4" />}
      title={t(I18nKey.AUTOMATIONS$DETAIL$PROMPT)}
    >
      <div className="flex flex-col gap-2">
        <div className="relative">
          <p
            ref={contentRef}
            className={cn(
              "whitespace-pre-wrap text-sm leading-6 text-content",
              isCollapsed &&
                cn(PROMPT_COLLAPSED_MAX_HEIGHT_CLASS, "overflow-hidden"),
            )}
            data-testid="automation-prompt-content"
          >
            {prompt}
          </p>
          {isCollapsed ? (
            <div
              className="pointer-events-none absolute inset-x-0 bottom-0 h-20 bg-gradient-to-t from-[var(--oh-surface)] to-transparent"
              aria-hidden
              data-testid="automation-prompt-fade"
            />
          ) : null}
        </div>

        {isOverflowing ? (
          <button
            type="button"
            className="self-start cursor-pointer text-sm font-normal text-[var(--oh-muted)] hover:text-white"
            onClick={() => setIsExpanded((expanded) => !expanded)}
            data-testid="automation-prompt-toggle"
          >
            {isExpanded
              ? t(I18nKey.COMMON$VIEW_LESS)
              : t(I18nKey.COMMON$VIEW_MORE)}
          </button>
        ) : null}
      </div>
    </SectionCard>
  );
}
