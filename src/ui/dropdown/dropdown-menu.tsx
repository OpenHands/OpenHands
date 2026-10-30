import React from "react";
import { Divider } from "#/ui/divider";
import { cn } from "#/utils/utils";
import { DropdownOption } from "./types";
import {
  dropdownMenuListClassName,
  dropdownMenuRowClassName,
  dropdownMenuRowIconWrapperClassName,
} from "#/utils/dropdown-classes";

interface DropdownMenuProps {
  isOpen: boolean;
  filteredOptions: DropdownOption[];
  selectedItem: DropdownOption | null;
  emptyMessage: string;
  getMenuProps: (props?: object) => object;
  getItemProps: (props: {
    item: DropdownOption;
    index: number;
    className?: string;
  }) => object;
  footer?: React.ReactNode;
  openUpward?: boolean;
  fitContent?: boolean;
  /** Accepted for API parity; this version already sizes non-fitContent menus to the trigger. */
  matchTriggerWidth?: boolean;
  tallMenu?: boolean;
}

export function DropdownMenu({
  isOpen,
  filteredOptions,
  selectedItem,
  emptyMessage,
  getMenuProps,
  getItemProps,
  footer,
  openUpward = false,
  fitContent = false,
  tallMenu = false,
}: DropdownMenuProps) {
  return (
    <div
      className={cn(
        "absolute z-50 overflow-hidden text-contrast",
        fitContent ? "min-w-full w-max" : "w-full",
        openUpward ? "bottom-full mb-1" : "mt-1",
        "bg-tertiary rounded-md context-menu-box-shadow p-1",
        tallMenu ? "max-h-96" : "max-h-60",
        "overflow-auto",
        !isOpen && "hidden",
      )}
    >
      <ul
        {...getMenuProps({ className: cn("p-0", dropdownMenuListClassName) })}
      >
        {isOpen && filteredOptions.length === 0 && (
          <li className="px-2 py-2 text-sm text-muted italic">
            {emptyMessage}
          </li>
        )}
        {isOpen &&
          filteredOptions.map((option, index) => (
            <React.Fragment key={option.value}>
              {option.group &&
              option.group !== filteredOptions[index - 1]?.group ? (
                <li
                  role="presentation"
                  className="truncate px-2 pt-2 pb-1 text-xs text-muted"
                >
                  {option.group}
                </li>
              ) : null}
              <li
                {...getItemProps({
                  item: option,
                  index,
                  className: cn(
                    dropdownMenuRowClassName,
                    "focus:outline-none",
                    selectedItem?.value === option.value &&
                      "bg-interactive-selected text-contrast",
                  ),
                })}
                // Grouped rows show a short label; keep the full one for screen readers.
                aria-label={option.group ? option.label : undefined}
              >
                {option.prefix ? (
                  <span className={dropdownMenuRowIconWrapperClassName}>
                    {option.prefix}
                  </span>
                ) : null}
                <span className="min-w-0 truncate" title={option.label}>
                  {option.group
                    ? (option.groupLabel ?? option.label)
                    : option.label}
                </span>
              </li>
            </React.Fragment>
          ))}
      </ul>
      {isOpen && footer ? (
        <>
          <Divider inset="menu" />
          <div className="p-0">{footer}</div>
        </>
      ) : null}
    </div>
  );
}
