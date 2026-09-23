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
}: DropdownMenuProps) {
  return (
    <div
      className={cn(
        "absolute z-50 overflow-hidden text-contrast",
        fitContent ? "min-w-full w-max" : "w-full",
        openUpward ? "bottom-full mb-1" : "mt-1",
        "bg-tertiary rounded-md context-menu-box-shadow p-1",
        "max-h-60 overflow-auto",
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
            <li
              key={option.value}
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
            >
              {option.prefix ? (
                <span className={dropdownMenuRowIconWrapperClassName}>
                  {option.prefix}
                </span>
              ) : null}
              <span className="min-w-0 truncate">{option.label}</span>
            </li>
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
