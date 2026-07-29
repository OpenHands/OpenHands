import React from "react";
import { Check, ChevronDown } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { cn } from "#/utils/utils";
import {
  dropdownFilterTriggerClassName,
  dropdownMenuListClassName,
  dropdownMenuRowClassName,
} from "#/utils/dropdown-classes";

interface EnumFilterDropdownProps<T extends string> {
  testId: string;
  value: T;
  onChange: (value: T) => void;
  options: readonly T[];
  labelKeyByValue: Record<T, I18nKey>;
}

export function EnumFilterDropdown<T extends string>({
  testId,
  value,
  onChange,
  options,
  labelKeyByValue,
}: EnumFilterDropdownProps<T>) {
  const { t } = useTranslation("openhands");
  const [open, setOpen] = React.useState(false);
  const containerRef = useClickOutsideElement<HTMLDivElement>(() =>
    setOpen(false),
  );

  const defaultOption = options[0];
  const selectedLabel = t(labelKeyByValue[value]);

  return (
    <div
      ref={containerRef}
      className="relative shrink-0 w-auto"
      data-testid={testId}
    >
      <button
        type="button"
        data-testid="dropdown-trigger"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={t(I18nKey.CONVERSATION_PANEL$FILTER_LABEL)}
        onClick={() => setOpen((prev) => !prev)}
        className={cn(
          dropdownFilterTriggerClassName,
          defaultOption &&
            value !== defaultOption &&
            "border-white/60 bg-white/10",
        )}
      >
        <span className="whitespace-nowrap">{selectedLabel}</span>
        <ChevronDown
          className={cn(
            "h-4 w-4 shrink-0 text-tertiary-alt transition-transform",
            open && "rotate-180",
          )}
          aria-hidden
        />
      </button>

      {open ? (
        <div
          role="menu"
          data-testid={`${testId}-menu`}
          aria-label={t(I18nKey.CONVERSATION_PANEL$FILTER_LABEL)}
          className={cn(
            "absolute right-0 top-full z-50 mt-1 min-w-full w-max",
            "max-h-60 overflow-auto rounded-[6px] bg-tertiary p-1 context-menu-box-shadow",
            dropdownMenuListClassName,
          )}
        >
          {options.map((option) => {
            const selected = option === value;
            return (
              <button
                key={option}
                type="button"
                role="menuitemradio"
                aria-checked={selected}
                data-testid={`${testId}-${option}`}
                onClick={() => {
                  onChange(option);
                  setOpen(false);
                }}
                className={cn(
                  dropdownMenuRowClassName,
                  selected && "bg-[var(--oh-interactive-selected)]",
                )}
              >
                <span className="min-w-0 flex-1 truncate">
                  {t(labelKeyByValue[option])}
                </span>
                {selected ? (
                  <Check className="h-4 w-4 shrink-0" aria-hidden />
                ) : null}
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
