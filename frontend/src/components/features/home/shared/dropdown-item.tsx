import React from "react";
import { cn } from "#/utils/utils";

interface DropdownItemProps<T> {
  item: T;
  index: number;
  isSelected: boolean;
  getItemProps: <Options>(options: any & Options) => any; // eslint-disable-line @typescript-eslint/no-explicit-any
  getDisplayText: (item: T) => string;
  getItemKey: (item: T) => string;
  isProviderDropdown?: boolean;
  renderIcon?: (item: T) => React.ReactNode;
  itemClassName?: string;
}

export function DropdownItem<T>({
  item,
  index,
  isSelected,
  getItemProps,
  getDisplayText,
  getItemKey,
  isProviderDropdown = false,
  renderIcon,
  itemClassName,
}: DropdownItemProps<T>) {
  const itemProps = getItemProps({
    index,
    item,
    className: cn(
      isProviderDropdown
        ? "px-2 py-0 cursor-pointer text-xs rounded-md mx-0 my-0 h-6 flex items-center"
        : "px-2 py-2 cursor-pointer text-sm rounded-md mx-0 my-0.5",
      "text-white focus:outline-none font-normal",
      {
        "bg-[#C9B974] text-black": isSelected,
        "hover:bg-[#5C5D62]": !isSelected,
        "hover:bg-[#C9B974] hover:text-black": isSelected,
      },
      itemClassName,
    ),
  });

  return (
    // eslint-disable-next-line react/jsx-props-no-spreading
    <li key={getItemKey(item)} {...itemProps}>
      <div className="flex items-center gap-2">
        {renderIcon?.(item)}
        <span className="font-medium">{getDisplayText(item)}</span>
      </div>
    </li>
  );
}
